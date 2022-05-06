from transformers import AdamW
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_curve,classification_report
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

######################################
# one train epoch                    #
######################################
def train(model, train_dataloader, device, cross_entropy):
    model.train()

    total_loss = 0
    # define the optimizer
    optimizer = AdamW(model.parameters(), lr = 1e-3)

    # empty list to save model predictions
    total_preds=[]

    # iterate over batches
    for step,batch in enumerate(train_dataloader):

        # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}'.format(step, len(train_dataloader)))
        
        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        sent_id, mask, label = batch
        # clear previously calculated gradients 
        model.zero_grad()  
        # get model predictions for the current batch
        preds = model(sent_id, mask)
        # compute the loss between actual and predicted values      
        loss = cross_entropy(preds, label)

        # add on to the total loss
        total_loss = total_loss + loss.item()
        # backward pass to calculate the gradients
        loss.backward() #GRADIENT
        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        # update parameters
        optimizer.step()
        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()
        # append the model predictions
        total_preds.append(preds)

    # compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)

    # predictions are in the form of (no. of batches, size of batch, no. of classes).
    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    #returns the loss and predictions
    return avg_loss, total_preds


# function for evaluating the model
def evaluate(model, val_dataloader, device, cross_entropy):
    print("Evaluating...")
    # deactivate dropout layers
    model.eval() #DROP OUT
    total_loss = 0
    # empty list to save the model predictions
    total_preds = []

    # iterate over batches
    for step,batch in enumerate(val_dataloader):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, label = batch
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)
            # compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,label)
            total_loss = total_loss + loss.item()
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    # compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds


#################################################
# train binary class for different question ids #
#################################################
def training(model, data, device, ep=10):
    # number of training epochs
    model.to(device)

    #compute the class weights
    class_wts = compute_class_weight('balanced', np.unique(data.train_y), data.train_y)
    # convert class weights to tensor
    weights= torch.tensor(class_wts,dtype=torch.float)
    weights = weights.to(device)
    # loss function
    cross_entropy  = nn.NLLLoss(weight=weights)

    # set initial loss to infinite
    best_valid_loss = float('inf')
    # empty lists to store training and validation loss of each epoch
    train_losses=[]
    valid_losses=[]
    #for each epoch
    epochs = int(ep)
    for epoch in range(epochs):
        print('\n ============Epoch {:} / {:}=============='.format(epoch + 1, epochs))
        #train model
        train_loss, _ = train(model, data.train_dataloader, device, cross_entropy)
        #evaluate model
        valid_loss, _ = evaluate(model, data.dev_dataloader, device, cross_entropy)

        #save the best model
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            print("SAVING MODEL")
            torch.save(model.state_dict(), './saved_model/'+model._get_name()+'.pt')
        # append training and validation loss
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print(f'Training Loss: {train_loss:.3f}')
        print(f'Validation Loss: {valid_loss:.3f}')


def testing(model, data, device):
    model.to(device)
    #load weights of best model
    path = './saved_model/'+model._get_name()+'.pt'
    model.load_state_dict(torch.load(path))

    total_preds = []
    # iterate over batches
    for step,batch in enumerate(data.test_dataloader):
        # push the batch to gpu
        batch = [t.to(device) for t in batch]
        sent_id, mask, label = batch
        # deactivate autograd
        with torch.no_grad():
            # model predictions
            preds = model(sent_id, mask)
            # compute the validation loss between actual and predicted values
            preds = preds.detach().cpu().numpy()
            total_preds.append(preds)

    # reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    # model's performance
    total_preds = np.argmax(total_preds, axis = 1)
    my_open = open('./result/'+model._get_name()+'.txt', 'a')
    #打开fie_name路径下的my_infor.txt文件,采用追加模式
    #若文件不存在,创建，若存在，追加
    report = classification_report(data.test_label, total_preds, digits=4)
    my_open.write('\n'+report)
    my_open.close()
    print(report)

    cm = confusion_matrix(y_true=data.test_label, y_pred=total_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig('./result/trainer.png')#保存图片






    