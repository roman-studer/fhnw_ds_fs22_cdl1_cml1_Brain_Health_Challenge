import pandas as pd
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader, ConcatDataset
import uuid
import sys
import optuna
from sklearn.model_selection import KFold
import torch.optim as optim

sys.path.insert(0, '../scripts/')
from helpers import miscellaneous as misc
from helpers import plotters as plot
from helpers import preprocessing2d as prep
from data_loader import get_data_loader
from loss_functions import get_optimizer, get_criterion
from ml_models import get_model as get_net



def nn_train(model, device, traindata, optimizer, criterion, epoch, scheduler, title):

    # test model
    test_loss = 0
    test_total = 0
    test_correct = 0

    y_true = []
    y_pred = []
    y_proba = None

    kfold = KFold(n_splits=CONFIG['K_FOLDS'], shuffle=True)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(traindata)):
        model.train()

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(traindata, batch_size=BATCH_SIZE, sampler=train_subsampler)
        test_loader = DataLoader(traindata, batch_size=BATCH_SIZE, sampler=test_subsampler)

        for batch_idx, (data, target) in enumerate(train_loader, start=0):
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            if CRITERION == 'Focal Loss':
                target_loss = torch.nn.functional.one_hot(target, num_classes=3)

                loss = criterion(output, target_loss)
            else:
                loss = criterion(output, target)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

        model.eval()
        for batch_idx, (data, target) in enumerate(test_loader, start=0):

            with torch.no_grad():
                data, target = data.to(device), target.to(device)

                outputs = model(data)
                if CRITERION == 'Focal Loss':
                    target_loss = torch.nn.functional.one_hot(target, num_classes=3)

                    loss = criterion(outputs, target_loss)
                else:
                    loss = criterion(outputs, target)

                test_loss += loss.item()

                scores, predictions = torch.max(outputs.data, 1)
                test_total += target.size(0)
                test_correct += int(sum(predictions == target))

                if y_proba is None:
                    y_proba = outputs.cpu()
                else:
                    y_proba = np.vstack((y_proba, outputs.cpu()))

                for i in target.tolist():
                    y_true.append(i)
                for j in predictions.tolist():
                    y_pred.append(j)

        scheduler.step()


    acc = round((test_correct / test_total) * 100, 2)

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': test_loss/ test_total,
        'acc': acc
    }, CONFIG['DATA_DIR_MODELS'] + title + '_' + str(epoch))

    print("Epoch [{}], Loss: {}, Accuracy: {}".format(epoch, test_loss / test_total, acc), end="")
    wandb.log({"train_loss_epoch": test_loss / test_total, "train_acc_epoch": acc, "Epoch": epoch,
               "learning_rate": scheduler.get_last_lr()[0]})

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
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = net(inputs)
            if CRITERION == 'Focal Loss':
                target_loss = torch.nn.functional.one_hot(labels, num_classes=3)

                loss = criterion(outputs, target_loss)
            else:
                loss = criterion(outputs, labels)

            test_loss += loss.item()

            scores, predictions = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += int(sum(predictions == labels))

            if y_proba is None:
                y_proba = outputs.cpu()
            else:
                y_proba = np.vstack((y_proba, outputs.cpu()))

            for i in labels.tolist():
                y_true.append(i)
            for j in predictions.tolist():
                y_pred.append(j)

    if return_prediction:
        return y_true, y_pred, y_proba

    acc = round((test_correct / test_total) * 100, 2)
    print(" Test_loss: {}, Test_accuracy: {}".format(test_loss / test_total, acc))
    wandb.log({"val_loss": test_loss / test_total, "val_acc": acc,
               })

    return acc


CONFIG = misc.get_config()

DEVICE = CONFIG['DEVICE']
LEARNING_RATE = CONFIG['LEARNING_RATE']
BATCH_SIZE = CONFIG['BATCH_SIZE']
EPOCHS = CONFIG['EPOCHS']
TRANSFORMER = CONFIG['TRANSFORMER']
NET = CONFIG['MODEL']
CRITERION = CONFIG['CRITERION']
OPTIMIZER = CONFIG['OPTIMIZER']
SAFE_MODEL = CONFIG['SAVE_MODEL']
SCHEDULER = CONFIG['SCHEDULER']
TRAIN_SET = '../' + CONFIG['TRAIN_LABELS_DIR']
TEST_SET = '../' + CONFIG['TEST_LABELS_DIR']
RAW_DATA = '../' + CONFIG['FLATTENED_DATA_DIR']
PLOT_DIR = '../' + CONFIG['PLOT_DIR_BINARY']
DIMENSION = CONFIG['DIMENSION']
NSLICE = CONFIG['NSLICE']
WANDB_USER = CONFIG['WANDB_USER']
DATA_LOADER = CONFIG['DATA_LOADER']




def objective(trial):
    ID = str(uuid.uuid4())
    NET_NAME = f'Net: {NET} Transf: {TRANSFORMER} Epochs: {EPOCHS}_ID:{ID}'


    LEARNING_RATE = trial.suggest_float('learning rate', 0.0001, 0.003)
    TITLE = CONFIG['NAME'] + f'_LR{LEARNING_RATE}_CR{CRITERION}_TR{TRANSFORMER}_OP{OPTIMIZER}_BS{BATCH_SIZE}_EP{EPOCHS}__ID_{ID}'

    wandb.init(project="mlmodels", entity="brain-health", dir='../models/',
               name=NET_NAME,
               settings=wandb.Settings(_disable_stats=True))

    wandb.define_metric("acc", summary="max")
    wandb.define_metric("loss", summary="min")

    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "transformer": TRANSFORMER,
        "net": NET,
        "criterion": CRITERION,
        "optimizer": OPTIMIZER,
        "image_size": CONFIG['IMAGE_RESIZE1']
    }

    class_names = ['CN', 'MCI', 'AD']


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using {device} for training')

    net = get_net().to(device)
    test_transform, train_transform = prep.get_transformer(TRANSFORMER)

    loader = get_data_loader()
    train_data = loader(TRAIN_SET, transform=train_transform, dimension=DIMENSION, nslice=NSLICE)
    test_data = loader(TEST_SET, transform=test_transform,  dimension=DIMENSION, nslice=NSLICE)

    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    criterion = get_criterion(CONFIG)
    optimizer = get_optimizer(net, CONFIG)

    if CONFIG['LOG_MODEL']:
        wandb.watch(net, log="parameters")

    if SCHEDULER == "ExponentialLR" or SCHEDULER is None:
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)


    print(f'[INFO] Optimizer: {OPTIMIZER}, learning rate: {LEARNING_RATE}, batch size: {BATCH_SIZE}')

    print('[INFO] Started Training')

    best_test_acc = 0

    for epoch in range(EPOCHS):
        nn_train(net, device, train_data, optimizer, criterion, epoch, scheduler, TITLE)
        test_acc = nn_test(net, device, test_dataloader, criterion, class_names)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc

        trial.report(test_acc, epoch)

        # Handle pruning based on the test accuracy
        #if trial.should_prune():
        #    raise optuna.TrialPruned()

    y_true, y_pred, y_proba = nn_test(net, device, test_dataloader, criterion, class_names, return_prediction=True)
    wandb.log(
        {#"roc": wandb.plot.roc_curve(np.array(y_true), np.array(y_proba), labels=class_names, classes_to_plot=None),
         "learning_rate": LEARNING_RATE,
         "epochs": EPOCHS,
         "batch_size": BATCH_SIZE,
         "transformer": TRANSFORMER,
         "net": NET,
         "criterion": CRITERION,
         "optimizer": OPTIMIZER,
         "image_size": CONFIG['IMAGE_RESIZE1'],
         "pretrained": CONFIG['PRETRAINED']})


    if SAFE_MODEL:
        print(f"[INFO] Saved model to models/ with name Net: {NET} Transf: {TRANSFORMER} Epochs: {EPOCHS}")
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': best_test_acc,
            }, f'../models/{TITLE}.pth')
        
    print("[INFO] Finished Training")
    wandb.finish()

    if device == 'cuda:o':
        torch.cuda.empty_cache()

    return best_test_acc


study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner)
study.optimize(objective, n_trials=CONFIG['N_TRIALS'])
