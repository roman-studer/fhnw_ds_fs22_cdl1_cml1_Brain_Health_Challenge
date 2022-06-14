import torch.nn as nn
import torch.optim as optim
from monai.losses import FocalLoss



def get_criterion(config):
    if config['CRITERION'] == 'CrossEntropy':
        return nn.CrossEntropyLoss()
    elif config['CRITERION'] == 'Focal Loss':
        return FocalLoss()
    
    else:
        print('[WARN] Unknown criterion selected')

def get_optimizer(net, config):
    if config['OPTIMIZER'] == 'SGD':
        return optim.SGD(net.parameters(), lr=config['LEARNING_RATE'], momentum=0.9)
    if config['OPTIMIZER'] == 'Adam':
        return optim.Adam(net.parameters(), lr=config['LEARNING_RATE'])
    if config['OPTIMIZER'] == 'AdamW':
        return optim.AdamW(net.parameters(), lr=config['LEARNING_RATE'], weight_decay=config['WEIGHT_DECAY'])
    
    else:
        print('[WARN] Unknown optimizter selected')
