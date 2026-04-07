import torch.optim.lr_scheduler as lr_scheduler

def build_scheduler(optimizer, config):
    """Learning rate scheduler for model plateau | step | cosine"""
    scheduler_name = config['training'].get('scheduler', 'reduce_lr_on_plateau')
    if scheduler_name == 'none':
        return None

    elif scheduler_name == 'reduce_lr_on_plateau':
        # reduce when val loss stopping reduce
        factor = config['training'].get('lr_factor', 0.5) # split a half when reduce
        patience = config['training'].get('lr_patience', 3) # after 3 epochs, loss not decrease -> split lr
        print(f"--> [Scheduler] ReduceLROnPlateau (factor={factor}, patience={patience})")
        
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
        )
    elif scheduler_name == 'step':
        # decay(decrease) every n epochs
        step_size = config['training'].get('lr_step_size', 10)  
        gamma = config['training'].get('lr_gamma', 0.1)         # Decrease 1/10
        print(f"--> [Scheduler] StepLR (step_size={step_size}, gamma={gamma})")
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == 'cosine':
        # decay with cosine
        T_max = config['training'].get('epochs', 101) 
        print(f"--> [Scheduler] CosineAnnealingLR (T_max={T_max})")
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    else:
        raise ValueError(f"Not supported this {scheduler_name} scheduler!") 
