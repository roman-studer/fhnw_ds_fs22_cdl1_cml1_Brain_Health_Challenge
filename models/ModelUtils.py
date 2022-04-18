import torch
import torch.nn as nn
import torchio as tio
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

#Add more parameters to Models so that it's 
class EncoderCNN(nn.Module):
    """
    Encodes a 2D image into output neurons
    Attributes
    ----------
    resnet: use resnet model structure
    linear: linear layer after resnet model
    bn: batchnormalisation layer definition
    ----------
    Methods
    ----------
    forward(images): takes images and does forward feed calculation
    """
    def __init__(self, CNN_out_size, train_CNN=False):
        """
        Parameters
        ----------
        CNN_out_size: int | Amount of ouput neurons
        train_CNN: boolean | use pretrained model yes/no
        """
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        
        #change to 3D conv net with nn.Conv3d()
        #resnet = models.resnet18(pretrained=True)
        #modules = list(resnet.children())[:-1]
        #self.resnet = nn.Sequential(*modules)
        
        self.convolute = nn.Conv3d(1, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))
        
        #self.linear = nn.Linear(resnet.fc.in_features, CNN_out_size)
        #replace hardcoded size to formula: 
        
        convolution_size = get_conv_output_shape(50, kernel_size=(3, 5, 2), stride=(2, 1, 1), pad=(4, 2, 0), dilation = 1, type_of_convolution = "3D")
        self.linear = nn.Linear(33 * np.prod(convolution_size), CNN_out_size)
        
        self.bn1 = nn.InstanceNorm3d(33 * np.prod(convolution_size))
        self.bn2 = nn.BatchNorm1d(CNN_out_size, momentum=0.01)
        
    def forward(self, images):
        """
        Parameters
        ----------
        images: torch.Tensor(batch size, _, _, _,) |  
        """
        
        #with torch.no_grad():
        #features = self.resnet(images)
        
        features = nn.functional.relu(self.convolute(images))
        features = self.bn1(features)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn2(features)
        features = torch.nn.functional.softmax(features, dim=1)
        return features
    
class CognitiveTestMLP(nn.Module):
    """
    Simple MLP for tabular data
    Attributes
    ----------
    layers: nn.Sequential object with MLP model setup
    ----------
    Methods
    ----------
    forward(features): takes label images and does forward feed calculation
    """
    def __init__(self, MLP_out, MLP_in):
        """
        Parameters
        ----------
        MLP_out: int | Amount of ouput neurons
        MLP_in: int | Amount of labels
        """
        super(CognitiveTestMLP, self).__init__()
        
        self.layers = nn.Sequential(
          nn.Linear(MLP_in, 64),
          nn.ReLU(),
          nn.Linear(64, 32),
          nn.ReLU(),
          nn.Linear(32, MLP_out)
        )
    
    def forward(self, features):
        return self.layers(features)

class FusionModel(nn.Module):
    """
    Fuses the CNN for images and MLP for labels together using a simple concatenate function
    Attributes
    ----------
    encoderCNN: EncoderCNN object
    cognitiveTestMLP: CognitiveTestMLP
    fusion: fusion layer
    fc: a final linear layer
    ----------
    Methods
    ----------
    forward(features): takes label images and does forward feed calculation
    """
    def __init__(self, CNN_out_size, MLP_out, MLP_in, fusion_output_size, num_classes):
        """
        Parameters
        ----------
        CNN_out_size: int | Amount of ouput neurons of CNN
        MLP_out: int | Amount of ouput neurons of MLP
        MLP_in: int | Amount of labels
        fusion_output_size: int | Amount of output neurons of fusion/concatenation layer
        num_classes: int | Amount of output classes/labels
        """
        super(FusionModel, self).__init__()
        self.encoderCNN = EncoderCNN(CNN_out_size)
        self.cognitiveTestMLP = CognitiveTestMLP(MLP_out, MLP_in)
        
        self.fusion = torch.nn.Linear(
            in_features=(MLP_out + CNN_out_size), 
            out_features=fusion_output_size
        )
        
        self.fc = torch.nn.Linear(
            in_features=fusion_output_size, 
            out_features=num_classes
        )
        
    def forward(self, images, test_data):
        """
        Data goes through the 2 early models then it gets put together and pushed through a linear layer and another final linear layer 
        Parameters
        ----------
        images: int | image as torch.Tensor
        test_data: int | tabular data
        """
        CNN_features = self.encoderCNN(images)
        MLP_features = self.cognitiveTestMLP(test_data)
        
        #here play around with squeeze() and unsqueeze() when using 3D images
        combined = torch.cat(
            [MLP_features.squeeze(1), CNN_features], dim=1
        )
        
        #here opportunity for another BN layer or 
        
        fused = torch.nn.functional.relu(self.fusion(combined))
        logits = self.fc(fused)
        pred = torch.nn.functional.softmax(logits)
        
        return pred
    
def get_conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1, type_of_convolution="2D"):
    """
    Utility function for computing output of convolutions
    takes a tuple of (h,w) and returns a tuple of (h,w)
    """
    
    if type_of_convolution == "3D":
        type_multiplicator = 3
    else:
        type_multiplicator = 2
    
    if type(h_w) is not tuple:
        h_w = (h_w,) * type_multiplicator
    
    if type(kernel_size) is not tuple:
        kernel_size = (kernel_size, ) * type_multiplicator
    
    if type(stride) is not tuple:
        stride = (stride, ) * type_multiplicator
    
    if type(pad) is not tuple:
        pad = (pad, ) * type_multiplicator
    
    value = []
    for i in range(0, type_multiplicator):
        value.append((h_w[i] + (2 * pad[i]) - (dilation * (kernel_size[i] - 1)) - 1)// stride[i] + 1)

    return value

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean().cpu())
            max_grads.append(p.grad.abs().max().cpu())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])