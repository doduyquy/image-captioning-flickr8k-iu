import os
import wandb
import torch
import argparse
from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger_wandb import init_wandb

from src.data.dataloader import get_loaders_flickr8k, get_loaders_iu_xray
from src.models import build_model
from src.training.trainer import Trainer
from src.training.losses import build_loss
from src.training.optimizer import build_optimizer
from src.training.scheduler import build_scheduler
from src.utils.checkpoint import load_checkpoints
from src.evaluation.evaluator import evaluate_model, evaluate_and_show
from src.utils.logger_wandb import save_model_to_wandb

from datetime import datetime
#-------------------------------------------------------------

def main():
    print("\t\t--> In main <--\t\t")

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
    print("--- Use device:", device)

    # get args 
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--dataset", type=str, default="flickr8k", choices=["flickr8k", "iu_xray"],
                        help="Dataset to train on: flickr8k | iu_xray")
    args = parser.parse_args()
    
    # load config
    config = load_config(args.config, args.env)
    set_seed(config['seed'].get('random_seed', 21))

    timestamp = datetime.now().strftime("%d%m%Y_%H%M")
    run_name = f"{config['model'].get('name', 'transformer')}_{args.dataset}_{timestamp}"

    # -----------------------------------------------------------------------
    # Data loading: switch giữa flickr8k và iu_xray
    # -----------------------------------------------------------------------
    if args.dataset == "flickr8k":
        data_path = config['flickr8k']['data_path']
        root_path = config['flickr8k']['root_path']

        captions_file_name = config["flickr8k"].get("captions_filename", "captions.txt")
        if not captions_file_name.endswith(".txt") and not captions_file_name.endswith(".csv"):
            captions_file_name += ".txt"

        image_dir = os.path.join(data_path, 'images')
        captions_file = os.path.join(data_path, captions_file_name)

        print(f"--> Image Dir: {image_dir}")
        print(f"--> Captions File: {captions_file}")

        loaders, vocab = get_loaders_flickr8k(
            data_dir=data_path,
            image_dir=image_dir,
            captions_file=captions_file,
            batch_size=config['data'].get('batch_size', 32),
            num_workers=config['data'].get('num_workers', 2),
            freq_threshold=config['data'].get('freq_threshold', 5)
        )

    elif args.dataset == "iu_xray":
        data_path = config['iu_xray']['data_path']
        root_path = config['iu_xray']['root_path']
        annotation_file = config['iu_xray']['annotation_file']
        image_dir = os.path.join(data_path, 'images')

        print(f"--> Data Path: {data_path}")
        print(f"--> Image Dir: {image_dir}")
        print(f"--> Annotation File: {annotation_file}")

        loaders, vocab = get_loaders_iu_xray(
            annotation_file=annotation_file,
            image_dir=image_dir,
            batch_size=config['data'].get('batch_size', 32),
            num_workers=config['data'].get('num_workers', 2),
            freq_threshold=config['data'].get('freq_threshold', 3)
        )

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    train_loader, val_loader, test_loader = loaders

    # Sử dụng build_model chuyên nghiệp
    model = build_model(config=config, vocab_size=len(vocab))
    
    # build loss & optimizer
    pad_idx = vocab.stoi["<pad>"]
    loss_name = config['training'].get('loss', 'cross_entropy')
    loss = build_loss(loss_name, pad_idx=pad_idx)

    optimizer = build_optimizer(mode_params=model.parameters(), config=config)
    scheduler = build_scheduler(optimizer=optimizer, config=config)
    
    # set path to save ckpt
    path_save_ckpt = os.path.join(root_path, f"outputs/checkpoints/{config['model'].get('name', 'transformer')}/{run_name}_best.pth")
    os.makedirs(os.path.dirname(path_save_ckpt), exist_ok=True)

    clip_grad_norm = config['training'].get('clip_grad_norm', 1.0)
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        vocab=vocab,
        criterion=loss,
        optimizer=optimizer,
        clip_grad_norm=clip_grad_norm,
        scheduler=scheduler,
        config=config,
        device=device,
        run_name=run_name,
        save_dir=path_save_ckpt
    )
    train_losses, val_losses, best_val_bleu4, best_epoch = trainer.fit()

    # evaluate
    print("\n" + "="*51)
    print("Evaluate on Test set")
    print("="*51)
    
    # Load lại checkpoint tốt nhất
    checkpoint = torch.load(path_save_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 1. Tính toán điểm số (BLEU-1, 2, 3, 4)
    test_metrics = evaluate_model(model, test_loader, vocab, device, method='beam')
    
    # Log metrics lên WandB nếu đang dùng
    if config['logging'].get('use_wandb', True):
        # Lưu lại các giá trị 'Best' vào summary để không bị lẫn với epoch cuối
        wandb.run.summary["best_val_bleu4"] = best_val_bleu4
        wandb.run.summary["best_epoch"]     = best_epoch
        
        # Chúng ta dùng log với tiền tố 'Test/' để phân biệt
        wandb.log({f"Test/{k}": v for k, v in test_metrics.items()})

    # 2. Hiển thị một số ví dụ trực quan
    evaluate_and_show(model, test_loader, vocab, device, method='beam', num_samples=10)
    
    # upload best ckpt to wandb
    if config['logging'].get('use_wandb', True):
        print("\n\t--> Uploading best ckpt to WandB, please wait...")
        save_model_to_wandb(path_save_ckpt)
        wandb.finish()

    print("\n\t\t--- ALL TASKS COMPLETED! ---\n")

    

if __name__ == "__main__":
    main()