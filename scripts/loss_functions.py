import torch.nn as nn
import torch.optim as optim



def get_criterion():
    return nn.CrossEntropyLoss()

def get_optimizer(net, config):
    return optim.SGD(net.parameters(), lr=config['LEARNING_RATE'], momentum=0.9)