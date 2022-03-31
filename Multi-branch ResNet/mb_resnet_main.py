import os
import sys
import h5py
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import itertools
from MB_ResNet_model import BasicBlock, Bottleneck, ResNet, resnet18,resnet34 

## argparse
parser = argparse.ArgumentParser(description='ResNet classification')
parser.add_argument('--cuda', action='store_true', help='Choose device to use cpu cuda')
parser.add_argument('--batch_size', action='store', type=int, 
                        default=4, help='number of data in a batch')
parser.add_argument('--lr', action='store', type=float, 
                        default=0.001, help='initial learning rate')
parser.add_argument('--epochs', action='store', type=int, 
                        default = 1, help='train rounds over training set')


## file
filename = './scsn_ps_2000_2017_shuf.hdf5'
## reading file
f = h5py.File(filename,'r') 

'''
an example
'''

## two classification: noise/non-noise
x_train_1 = f['X']
y_train_1 = f['Y']
yy_train = [1]*y_train_1.shape[0]
for i in range(y_train_1.shape[0]):
    if f['Y'][i] ==0 or f['Y'][i] == 1:
        yy_train[i] = 0
yy_train = np.array(yy_train)
x_train_coarse = x_train_1[:800,:]
y_train_coarse = yy_train[:800]

x_test_coarse = x_train_1[800:1000,:]
y_test_coarse = yy_train[800:1000]

## 3 classification: P/S
x_train_fine = f['X'][:800,:]
y_train_fine = f['Y'][:1000] 

x_test_fine  = f['X'][800:1000,:]
y_test_fine  = f['Y'][800:1000]

## dataload
class get_dataset(Dataset):
    def __init__(self, x, y, z):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.z = torch.from_numpy(z)
    def __getitem__(self, index):
        return (self.x[index], self.y[index], self.z[index])
    def __len__(self):
        return len(self.y)

train_dataset = get_dataset(x_train_coarse, y_train_coarse, y_train_fine)
test_dataset  = get_dataset(x_test_coarse, y_test_coarse,y_test_fine )

def main(arngs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader  = DataLoader(dataset = test_dataset,  batch_size = args.batch_size, shuffle = False)
    
    model = resnet34(args).to(device)
    criterion_coarse = nn.CrossEntropyLoss()
    criterion_fine = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    ## Train the model
    train_loss1 = []
    train_acc1 = []
    train_loss2 = []
    train_acc2 = []
    train_loss_final =[]
    best_accuracy = 0.9
    for epoch in range(args.epochs):
        batch_loss1 = []
        batch_accuracy1 =[]
        batch_loss2 = []
        batch_accuracy2 =[]
        batch_loss_final = []
        model.train()
        for step, (images, labels_coarse, labels_fine) in enumerate(train_loader):
            labels_coarse = labels_coarse.to(device)
            labels_fine   = labels_fine.to(device)
            
            outputs_coarse, outputs_fine = model(torch.transpose(images,2,1).to(device))  # transpose
            loss_coarse = criterion_coarse(outputs_coarse, labels_coarse.long())
            batch_loss1.append(loss_coarse.item())
            loss_fine = criterion_fine(outputs_fine, labels_fine.long())
            batch_loss2.append(loss_fine.item())
            
            loss = 0.5 * loss_coarse + 0.5*loss_fine
            batch_loss_final.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            ## level 1 classification
            total_coarse = labels_coarse.size(0)
            pred_coarse = outputs_coarse.argmax(dim =1)
            correct_coarse = torch.eq(pred_coarse, labels_coarse).sum().item()
            accuracy_coarse = correct_coarse/total_coarse
            batch_accuracy1.append(accuracy_coarse)
            
            ## level 2  classification
            total_fine = labels_fine.size(0)
            pred_fine = outputs_fine.argmax(dim =1)
            correct_fine = torch.eq(pred_fine, labels_fine).sum().item()
            accuracy_fine = correct_fine/total_fine
            batch_accuracy2.append(accuracy_fine)
            
            if step % 10 == 0:
                print('Epoch  [{}/{}], Loss= {:.4f}, Accuracy_coarse = {:.4f}, Accuracy_fine = {:.4f}'.format(
                        epoch + 1, args.epochs,loss.item(), accuracy_coarse, accuracy_fine))
                
        train_loss1.append(sum(batch_loss1)/len(batch_loss1))
        train_acc1.append(sum(batch_accuracy1)/len(batch_accuracy1))
        train_loss2.append(sum(batch_loss2)/len(batch_loss2))
        train_acc2.append(sum(batch_accuracy2)/len(batch_accuracy2))
        train_loss_final.append(sum(batch_loss_final)/len(batch_loss_final))
        
        if sum(batch_accuracy2)/len(batch_accuracy2) > best_accuracy:
            best_accuracy = sum(batch_accuracy2)/len(batch_accuracy2)
            model_name = 'time_ResNet_parameter_'+str(sum(batch_accuracy2)/len(batch_accuracy2))
            torch.save(model.state_dict(), './' + model_name + '.pkl')
   
    plt.figure()
    plt.plot(train_acc1, '.-r', label = 'Coarse')
    plt.plot(train_acc2, '.-b', label = 'Fine')
    plt.title('Training Accuracy', fontsize = 12)
    plt.xlabel("Epochs", fontsize = 12)
    plt.savefig('./train_accuracy_Resnet.jpeg', dpi = 600)
    plt.close()

    plt.figure()
    plt.plot(train_loss1, '.-r', label = 'Coarse')
    plt.plot(train_loss2, '.-b', label = 'Fine')
    plt.plot(train_loss_final, '.-k', label = 'Final')
    plt.title('Training Loss', fontsize = 12)
    plt.xlabel("Epochs", fontsize = 12)
    plt.savefig('./train_loss_Resnet.jpeg', dpi = 600)
    plt.close()
    
    torch.save(model.state_dict(), './mb_resnet_model.pkl')

    ## Test the model
    model.eval()
    out_coarse = []
    test_loss_coarse = []
    test_accuracy_coarse = []
    
    out_fine = []
    test_loss_fine = []
    test_accuracy_fine = []
    
    test_loss_final = []

    with torch.no_grad():
        correct_coarse = 0
        total_coarse = 0
        correct_fine = 0
        total_fine = 0
        for images, labels_coarse, labels_fine in test_loader:
            outputs_coarse, outputs_fine = model(torch.transpose(images,2,1).to(device))
            ## classification 1
            _, pred_coarse = torch.max(outputs_coarse.data, 1)
            labels_coarse = labels_coarse.to(device)
            loss_coarse = criterion_coarse(outputs_coarse, labels_coarse.long())
            test_loss_coarse.append(loss_coarse.item())
            total_coarse += labels_coarse.size(0)
            correct_coarse += (pred_coarse.cpu()== labels_coarse.cpu()).sum().item()
            out_coarse.append(pred_coarse.cpu())
            accuracy_coarse = (pred_coarse.cpu() == labels_coarse.cpu()).sum().item() / labels_coarse.cpu().size(0)
            test_accuracy_coarse.append(accuracy_coarse)
            print('Batch Coarse Test Accuracy = ', accuracy_coarse)
            
            ## classification 2
            _, pred_fine = torch.max(outputs_fine.data, 1)
            labels_fine = labels_fine.to(device)
            loss_fine = criterion_fine(outputs_fine, labels_fine.long())
            test_loss_fine.append(loss_fine.item())
            total_fine += labels_fine.size(0)
            correct_fine += (pred_fine.cpu()== labels_fine.cpu()).sum().item()
            out_fine.append(pred_fine.cpu())
            accuracy_fine = (pred_fine.cpu() == labels_fine.cpu()).sum().item() / labels_fine.cpu().size(0)
            test_accuracy_fine.append(accuracy_fine)
            print('Batch Fine Test Accuracy = ', accuracy_fine)
            
            loss = 0.5*loss_coarse + 0.5*loss_fine
            test_loss_final.append(loss.item())
    
    print('Coarse Test Accuracy = ', correct_coarse / total_coarse)
    print('Fine Test Accuracy = ', correct_fine/total_fine)
    

    out_coarse = torch.cat(out_coarse, dim=0)
    pre_coarse = out_coarse.detach().numpy()
    
    ## confusion matrix plotting
    cm1 = confusion_matrix(y_test_coarse.tolist(), pre_coarse)
    multi_label1 = ['Noise', 'Non-noise']
    tick_marks = np.arange(len(multi_label1))
    plt.xticks(tick_marks, multi_label1, rotation=45)
    plt.yticks(tick_marks, multi_label1)
    thresh = cm1.max() / 2
    for i, j in itertools.product(range(cm1.shape[0]), range(cm1.shape[1])):
        plt.text(j, i, "{:,}".format(cm1[i, j]),horizontalalignment="center",
            color="black" if cm1[i, j] > thresh else "black",
            fontsize = 12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.imshow(cm1, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Confusion Matrix of Noise and Non-noise Detection', fontsize = 12)
    plt.colorbar()
    plt.savefig('./confusion_matrix_coarse.jpeg', dpi = 600)
    plt.close()
    
    print(metrics.classification_report(y_test_coarse.tolist(), pre_coarse, target_names = multi_label1, digits=6))
    
    out_fine = torch.cat(out_fine, dim=0)
    pre_fine = out_fine.detach().numpy()

    cm2 = confusion_matrix(y_test_fine.tolist(), pre_fine)
    multi_label2 = ['P', 'S', 'Noise']
    tick_marks = np.arange(len(multi_label2))
    plt.xticks(tick_marks, multi_label2, rotation=45)
    plt.yticks(tick_marks, multi_label2)
    thresh = cm2.max() / 2
    for i, j in itertools.product(range(cm2.shape[0]), range(cm2.shape[1])):
        plt.text(j, i, "{:,}".format(cm2[i, j]),horizontalalignment="center",
            color="black" if cm2[i, j] > thresh else "black",
            fontsize = 12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.imshow(cm2, interpolation='nearest', cmap=plt.cm.GnBu)
    plt.title('Confusion Matrix on Phase Detection', fontsize = 12)
    plt.colorbar()
    plt.savefig('./confusion_matrix_fine.jpeg', dpi = 600)
    plt.close()
    
    print(metrics.classification_report(y_test_fine.tolist(), pre_fine, target_names = multi_label2, digits=6))

if __name__ == "__main__":
    args = parser.parse_args() # Namespace object
    main(args)
