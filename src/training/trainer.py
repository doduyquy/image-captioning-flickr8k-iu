import torch
from torch import device
from torch.nn.utils import clip_grad_norm_
import os
import numpy as np 
from datetime import datetime
from src.utils.logger_wandb import init_wandb, log_metrics

class Trainer:
    """Forward -> Compute loss -> zero_grad -> Backward -> Update weights (step)"""
    def __init__(self, model, train_loader, val_loader, vocab, criterion, optimizer, clip_grad_norm, scheduler, config, device, run_name, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.clip = clip_grad_norm
        self.scheduler = scheduler
        self.device = device
        self.epochs = config['training'].get('epochs', 100)
        self.patience = config['training'].get('patience', 20)
        self.model_name = config['model'].get('name', 'lstm')
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.run_name = run_name
        self.config = config
        self.path_save_ckpt = save_dir
    

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0

        for images, captions, _ in self.train_loader:
            images = images.to(self.device)
            captions = captions.to(self.device)

            # shift caption
            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = self.model(images, inputs)

            loss = self.criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                targets.reshape(-1)
            )

            self.optimizer.zero_grad()
            loss.backward()

            clip_grad_norm_(self.model.parameters(), self.clip)

            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(self.train_loader)

    @torch.no_grad()
    def validate_one_epoch(self):
        self.model.eval()
        total_loss = 0

        for images, captions, _ in self.val_loader:
            images = images.to(self.device)
            captions = captions.to(self.device)

            inputs = captions[:, :-1]
            targets = captions[:, 1:]

            outputs = self.model(images, inputs)

            loss = self.criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                targets.reshape(-1)
            )

            total_loss += loss.item()

        return total_loss / len(self.val_loader)



    def fit(self):
        """ Fit your model
        Return:
            all_train_loss, all_val_loss
        """
        print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')

        if self.use_wandb:
            init_wandb(config=self.config, run_name=self.run_name)

        best_val_loss = float("inf")
        patience_counter = 0
        all_train_loss = []
        all_val_loss = []

        print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        for ep in range(self.epochs):

            train_loss = self.train_one_epoch()
            val_loss = self.validate_one_epoch()

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            print(
                f"Epoch {ep+1}/{self.epochs} - "
                f"loss: {train_loss:.4f} - "
                f"val_loss: {val_loss:.4f}"
            )

            # wandb log
            if self.use_wandb:
                log_metrics({
                    "Epoch": ep + 1,
                    "Train/Loss": train_loss,
                    "Val/Loss": val_loss,
                    "Learning_Rate": self.optimizer.param_groups[0]['lr']
                }, epoch=ep)

            # lr scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # save checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": ep
                }, self.path_save_ckpt)
                print(f"\t--- Save best at ep {ep+1}, val_loss: {val_loss:.4f}, path: {self.path_save_ckpt} ---")

            else:
                patience_counter += 1
                print(f"\t-!- No improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping at ep={ep+1}")
                    break

        return all_train_loss, all_val_loss



if __name__ == "__main__":
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn
    
    print("Test training...")

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 7)
        def forward(self, x):
            return self.fc(x)

    class DummyDataset(Dataset):
        def __len__(self): return 16
        def __getitem__(self, idx):
            return torch.randn(10), torch.randint(0, 7, (1,)).item()

    mock_config = {
        'training': {'epochs': 3, 'patience': 2},
        'path': {'root': '/tmp/'},
        'model': {'name': 'dummy_model'}
    }

    train_loader = DataLoader(DummyDataset(), batch_size=8)
    val_loader = DataLoader(DummyDataset(), batch_size=8)

    model = DummyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    try:
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, mock_config, device)
        print("Fitting...")
        trainer.fit()
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")