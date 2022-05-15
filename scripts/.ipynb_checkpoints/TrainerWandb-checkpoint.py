import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
import torch.nn.functional as F

import sys

sys.path.insert(0, '../scripts/')
from helpers import miscellaneous as misc
from helpers import preprocessing2d as prep
from data_loader import MRIDataset
from loss_functions import get_optimizer, get_criterion
import monai

from sklearn.metrics import roc_curve

# import model named as Net
#from ml_models import LeNet as Net 
#NET = 'LeNet'

def nn_train(model, device, train_dataloader, optimizer, criterion, epoch, steps_per_epoch=20):
    model.train()

    train_loss = 0
    train_total = 0
    train_correct = 0

    for batch_idx, data in enumerate(train_dataloader, start=0):
        data, target = data['images'].to(device), data['labels'].to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        train_loss += loss.item()
        
        scores, predictions = torch.max(output.data, 1)
        train_total += target.size(0)
        predictions = F.one_hot(predictions, num_classes=3)
        train_correct += (predictions*target).sum()
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
    acc = round(((train_correct / train_total) * 100).item(), 3)
    print("Epoch [{}], Loss: {}, Accuracy: {}".format(epoch, train_loss / train_total, acc), end="")
    wandb.log({"train_acc": acc})
    wandb.log({"train_loss": train_loss})
    wandb.log({"train_loss_ma": train_loss/train_total})

    return None


def nn_test(net, device, test_dataloader, criterion, classes, return_prediction=False):
    net.eval()

    # test model
    test_loss = 0
    test_total = 0
    test_correct = 0

    y_true = []
    y_pred = []
    y_proba = None

    with torch.no_grad():
        for data in test_dataloader:
            inputs, target = data['images'].to(device), data['labels'].to(device)

            outputs = net(inputs)
            test_loss += criterion(outputs, target).item()

            scores, predictions = torch.max(outputs.data, 1)
            predictions = F.one_hot(predictions, num_classes=3)
            test_total += target.size(0)
            test_correct += (predictions*target).sum()

            if y_proba is None:
                y_proba = outputs
            else:
                y_proba = np.vstack((y_proba.cpu(), outputs.cpu()))

            for i in target.tolist():
                y_true.append(i)
            for j in predictions.tolist():
                y_pred.append(j)

    if return_prediction:
        return y_true, y_pred, y_proba

    acc = round(((test_correct / test_total) * 100).item(), 3)
    print(" Test_loss: {}, Test_accuracy: {}".format(test_loss / test_total, acc))
    wandb.log({"test_acc": acc})
    wandb.log({"test_loss": test_loss})
    wandb.log({"test_loss_ma": test_loss/test_total})

    return None

CONFIG = misc.get_config()

DEVICE = CONFIG['DEVICE']
LEARNING_RATE = CONFIG['LEARNING_RATE']
BATCH_SIZE = 256#CONFIG['BATCH_SIZE']
EPOCHS = 30#CONFIG['EPOCHS']
TRANSFORMER = CONFIG['TRANSFORMER']
CRITERION = CONFIG['CRITERION']
OPTIMIZER = CONFIG['OPTIMIZER']
TRAIN_SET = '../' + CONFIG['TRAIN_LABELS_DIR']
TEST_SET = '../' + CONFIG['TEST_LABELS_DIR']
RAW_DATA = '../' + CONFIG['FLATTENED_DATA_DIR']
PLOT_DIR = '../' + CONFIG['PLOT_DIR_BINARY']
DIMENSION = CONFIG['DIMENSION']
NSLICE = CONFIG['NSLICE']
WANDB_USER = CONFIG['WANDB_USER']

conv = monai.networks.blocks.Convolution(
    dimensions=2,
    in_channels=1,
    out_channels=1,
    adn_ordering="ADN",
    act=("prelu", {"init": 0.2}),
    dropout=0.1
)

cnn_to_mlp = torch.nn.Sequential(
  torch.nn.Flatten(1, -1),
  torch.nn.Linear(150*150, 32),
  torch.nn.ReLU(),
  torch.nn.Linear(32, 3),  
  torch.nn.Softmax(dim=1)
)

NET = torch.nn.Sequential(
    conv,
    cnn_to_mlp
)

wandb.init(project="mlmodels", entity="brain-health",
          settings=wandb.Settings(_disable_stats=True))

# wandb.init(project="mlmodels", entity=WANDB_USER,
#          name=f'Net: {NET} Transf: {TRANSFORMER} Epochs: {EPOCHS}')

wandb.define_metric("acc", summary="max")
wandb.define_metric("loss", summary="min")

wandb.config = {
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "transformer": TRANSFORMER,
    "net": NET,
    "criterion": CRITERION,
    "optimizer": OPTIMIZER
}


class_names = ['CN', 'MCI', 'AD']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = NET.to(device)
test_transform, train_transform = prep.get_transformer(TRANSFORMER)

train_data = MRIDataset(dataset_path=TRAIN_SET, transform=train_transform, dimension=DIMENSION, nslice=NSLICE)
test_data = MRIDataset(dataset_path=TEST_SET, transform=test_transform, dimension=DIMENSION, nslice=NSLICE)

train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

criterion = get_criterion()
optimizer = get_optimizer(net, CONFIG)

wandb.watch(net, log="parameters")

print("[INFO] Started training")
for epoch in range(EPOCHS):
    nn_train(net, device, train_dataloader, optimizer, criterion, epoch)
    nn_test(net, device, test_dataloader, criterion, class_names)


y_true, y_pred, y_proba = nn_test(net, device, test_dataloader, criterion, class_names, return_prediction=True)
    
y_true = torch.as_tensor(y_true)
y_pred = torch.as_tensor(y_pred)

print(np.array(torch.max(y_true,1)[1].cpu()), np.array(torch.max(y_pred,1)[1].cpu()))

#roc throws "IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed"
wandb.log(
    {#"roc": wandb.plot.roc_curve(y, scores, labels=class_names, classes_to_plot=None), 
     "learning_rate": LEARNING_RATE,
     "epochs": EPOCHS,
     "batch_size": BATCH_SIZE,
     "transformer": TRANSFORMER,
     "net": NET,
     "criterion": CRITERION,
     "optimizer": OPTIMIZER})

print("[INFO] Finished Training")
wandb.finish()
