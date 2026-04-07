import torch.optim as optim

def build_optimizer(mode_params, config):

    
    train_cfg = config.get('training', {})
    opt_name = train_cfg.get('optimizer', 'adam').lower()
    lr = train_cfg.get('lr', 0.001)
    weight_decay = train_cfg.get('weight_decay', 0.0001)


    if opt_name == 'adam':
        return optim.Adam(mode_params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        gamma = train_cfg.get('gamma', 0.9) 
        return optim.SGD(mode_params, lr=lr, weight_decay=weight_decay, momentum=gamma)
    # add another optimizer
    else:
        raise ValueError(f"Optimizer {opt_name} unsupported!")
