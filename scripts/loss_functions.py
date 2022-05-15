import torch.nn as nn
import torch.optim as optim



def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(net, config):
    if config['OPTIMIZER'] == 'SGD':
        return optim.SGD
    elif config['OPTIMIZER'] == 'Adam':
        return optim.Adam
    elif config['OPTIMIZER'] == 'AdamW':
        return optim.AdamW
    
    else:
        print('[WARN] Unknown optimizter selected')
