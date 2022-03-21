import torch
import torch.nn as nn
import torchio as tio

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
        resnet = models.resnet18(pretrained=True)
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        self.linear = nn.Linear(resnet.fc.in_features, CNN_out_size)
        
        self.bn = nn.BatchNorm1d(CNN_out_size, momentum=0.01)
        
    def forward(self, images):
        """
        Parameters
        ----------
        images: torch.Tensor(batch size, _, _, _,) |  
        """
        
        #with torch.no_grad():
        #    features = self.resnet(images)
        
        features = self.convolute(images)
        
        features = features.reshape(features.size(0), -1)
        features = self.bn(self.linear(features))
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