import torch.nn as nn


def build_loss(loss_name, pad_idx=None):
    """ Define loss for traning, cross_entropy: default
        Args:
            config: all config load from yaml
    """
    if loss_name == 'cross_entropy':
        loss = nn.CrossEntropyLoss(ignore_index=pad_idx)
    
    else: 
        raise ValueError(f"\n[!!!] Not support {loss_name} loss!\n")

    return loss