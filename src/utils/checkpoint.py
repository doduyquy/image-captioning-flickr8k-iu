import torch
import os

def save_checkpoint():
    pass

def load_checkpoints(model, optimizer, checkpoint_path, device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Not found file {checkpoint_path}")

    print(f"--> Loading ckpt from {checkpoint_path}")

    ckpt = torch.load(checkpoint_path)
    # load weight -> model
    model.load_state_dict(ckpt['model_state_dict'])
    # load optimizer and return current checkpoint
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    return ckpt['epoch']


def _save_checkpoint(path, model, optimizer, scheduler, epoch, config):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if scheduler else None,
            "config": config,
        },
        path,
    )


def _load_checkpoint(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and ckpt.get("scheduler_state") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    start_epoch = int(ckpt.get("epoch", -1)) + 1
    return start_epoch


def _log_sample_images(run, loader, step, max_images=4):
    if run is None or loader is None:
        return
    try:
        images, _ = next(iter(loader))
        n = min(max_images, images.size(0))
        preview = []
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        for i in range(n):
            # Convert CHW tensor to HWC for WandB image logging.
            img = images[i].detach().cpu()
            img = img * std + mean  # de-normalize to viewable RGB range
            img = img.permute(1, 2, 0)
            img = torch.clamp(img, 0.0, 1.0).numpy()
            preview.append(wandb.Image(img, caption=f"sample_{i}"))
        wandb.log({"train_samples": preview}, step=step)
    except Exception as e:
        print(f"[wandb] Skip image logging: {e}")


def _upload_checkpoint_artifact(run, ckpt_path, epoch):
    if run is None:
        return
    try:
        artifact = wandb.Artifact(name=f"checkpoint-epoch-{epoch+1}", type="model")
        artifact.add_file(ckpt_path)
        run.log_artifact(artifact, aliases=["latest", f"epoch_{epoch+1}"])
    except Exception as e:
        print(f"[wandb] Skip checkpoint artifact upload: {e}")
