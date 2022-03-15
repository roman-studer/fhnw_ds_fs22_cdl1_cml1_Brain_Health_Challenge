import torch
import numpy as np
import pandas as pd

# Define Train Function
def train(train_loader, model, optimizer, max_epoch, device, summary_writer = None, val_loader = None):
    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache
    
    step = 0
    
    loss_fn = nn.CrossEntropyLoss()
    
    # Set Trackers
    epochs_list = []
    train_loss_list = []
    train_loss_val = []
    val_loss_list = []
    eta_list = []
    
    for epoch in range(max_epochs):
        print("start epoch")
        
        #more trackers
        correct = 0
        total_labels = 0
        total_loss_val = 0
        
        t0 = time.time()
        
        #train
        total_loss = feed_train(model, loss_fn, optimizer, device, train_loader)
        print("Train Epoch Finished")
        avg_total_loss = total_loss / (idx + 1)
        
        #val
        if (val_loader is not None):
            total_loss_val = feed_val(model, loss_fn, device, val_loader)
            avg_total_loss_val = total_loss_val / (idx + 1)
            print("Validation Epoch Finished")
        else:
            avg_total_loss_val = 0
        
        t1 = time.time()
        eta = t1-t0
        eta_list.append(eta)
            
        print(f'Epoch: [{epoch+1}/{max_epochs}], average total train loss: {avg_total_loss}, train epoch time (s): {eta_list[-1:]}, average validation loss: {avg_total_loss_val}')
        epochs_list.append(epoch+1)
        train_loss_list.append(avg_total_loss)
        train_loss_val.append(avg_total_loss_val)
        if(summary_writer is not None):  
            summary_writer.add_scalar("Training/Loss", avg_total_loss)
            summary_writer.add_scalar("Val/Loss", avg_total_loss_val)
    
    metrics_tracker = [{'epoch': epochs_list[i], 'train_loss': train_loss_list[i], 'training_time': eta_list[i], 'val_loss': train_loss_val[i]} for i in range(max_epochs)]
    torch.cuda.synchronize()  
    torch.cuda.empty_cache()
    return pd.DataFrame(metrics_tracker)

def feed_train(model, loss_fn, optimizer, train_loader)
    # Per Epoch: Train Model 
    for idx, (imgs, test_results, captions) in enumerate(train_loader):
        with torch.autocast():
            #set model to train mode
            model.train()

            imgs = imgs.to(device)
            captions = captions.to(device)

            #send all captions but last one so that it learns to predict the end token 
            outputs = model(imgs, test_results)
            print(imgs.shape, test_results.shape)
            loss = loss_fn(outputs, captions)

            #loss gets updated after each batch, so a total loss is better to see if model is improving
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        #uncomment for more memory if needed
        #torch.cuda.empty_cache
            
    return total_loss

def feed_val(model, loss_fn, val_loader)
    # Per Epoch: Train Model   
    for idx, (imgs, test_results, captions) in enumerate(val_loader):
        with torch.autocast():
            #set model to train mode
            model.eval()

            imgs = imgs.to(device)
            captions = captions.to(device)

            #send all captions but last one so that it learns to predict the end token 
            outputs = model(imgs, test_results)
            loss_val = loss_fn(outputs, captions)

            #loss gets updated after each batch, so a total loss is better to see if model is improving
            total_loss_val += loss_val.item()
            
        #uncomment for more memory if needed
        #torch.cuda.empty_cache
        
    return total_loss_val