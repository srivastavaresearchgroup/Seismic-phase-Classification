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
from ResNet_model import BasicBlock
from ResNet_model import Bottleneck
from ResNet_model import ResNet
from ResNet_model import resnet18,resnet34,resnet50,resnet101,resnet152,resnext50_32x4d,resnext101_32x8d
from scipy.special import softmax

parser = argparse.ArgumentParser(description='ResNet classification')
parser.add_argument('--cuda', action='store_true', help='Choose device to use cpu cuda')
parser.add_argument('--batch_size', action='store', type=int, 
                        default=4, help='number of data in a batch')
parser.add_argument('--lr', action='store', type=float, 
                        default=0.001, help='initial learning rate')
parser.add_argument('--epochs', action='store', type=int, 
                        default = 1, help='train rounds over training set')
parser.add_argument('--num_classes', action='store', type=int, 
                        default = 3, help='total number of classes in dataset')

filename = './scsn_ps_2000_2017_shuf.hdf5'
# reading file
f = h5py.File(filename,'r') 

# example
train_x = f['X'][:800,:]# dimension: (4773750, 400, 3)
train_y = f['Y'][:800] # 0,1,2, shape: (4773750,)

test_x = f['X'][800:1000,:]
test_y = f['Y'][800::1000]

# dataloader
class get_dataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    def __len__(self):
        return len(self.y)

train_tensor_dataset = get_dataset(train_x, train_y)
test_tensor_dataset  = get_dataset(test_x, test_y)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(dataset = train_tensor_dataset, batch_size = args.batch_size, shuffle = True)
    test_loader  = DataLoader(dataset = test_tensor_dataset,  batch_size = args.batch_size, shuffle = False)

    model = resnet34(args).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    
    ## model training
    train_loss = []
    train_acc = []
    for epoch in range(args.epochs):
        batch_loss = []
        batch_acc =[]
        model.train()
        for step, (images, labels) in enumerate(train_loader):
            labels = labels.to(device)
            outputs = model(torch.transpose(images,2,1).to(device))   
            loss = criterion(outputs, labels.long())
            batch_loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total = labels.size(0)
            predicted = outputs.argmax(dim =1)
            correct = torch.eq(predicted, labels).sum().item()
            acc = correct/total
            batch_acc.append(acc)

            if step % 10 == 0:
                print('Epoch  [{}/{}], Loss= {:.4f}, Accuracy = {:.4f}'.format(epoch + 1, args.epochs,loss.item(), acc))
    
        train_loss.append(sum(batch_loss)/len(batch_loss))
        train_acc.append(sum(batch_acc)/len(batch_acc))
    
    # write csv file
    train_loss_dataframe = pd.DataFrame(data = train_loss)
    train_acc_dataframe = pd.DataFrame( data = train_acc)
    train_loss_dataframe.to_csv('./train_loss.csv',index=False, header = False)
    train_acc_dataframe.to_csv('./train_accuracy.csv',index=False, header = False)

    fig = plt.figure()
    plt.plot(train_acc, '.-', label = 'Training accuracy')
    plt.title('Training Accuracy')
    plt.xlabel("Epochs")
    plt.savefig('./train_accuracy.jpeg', dpi = 600)
    plt.close()

    plt.figure()
    plt.plot(train_loss, '.-', label ='Training loss')
    plt.title('Training loss')
    plt.xlabel("Epochs")
    plt.savefig('./train_loss.jpeg', dpi = 600)
    plt.close()
    
    # model saving
    torch.save(model.state_dict(), './model.pkl')

    # model testing
    model.eval()
    out = []
    test_loss = []
    test_acc = []

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = model(torch.transpose(images,2,1).to(device))
            _, predicted = torch.max(outputs.data, 1)
            labels = labels.to(device)
            loss = criterion(outputs, labels.long())
            test_loss.append(loss.item())
            total += labels.size(0)
            correct += (predicted.cpu()== labels.cpu()).sum().item()
            out.append(predicted.cpu())
            acc = (predicted.cpu() == labels.cpu()).sum().item() / labels.cpu().size(0)
            test_acc.append(acc)
            print('Batch Test Accuracy = ', acc)
    
   print('Test Accuracy = {} %'.format((correct / total) * 100))
   test_loss_dataframe = pd.DataFrame( data = test_loss)
   test_acc_dataframe = pd.DataFrame(data = test_acc)
   test_loss_dataframe.to_csv('./test_loss.csv',index=False, header=False)
   test_acc_dataframe.to_csv('./test_accuracy.csv',index=False, header = False)

   out = torch.cat(out,dim=0)
   pre = out.detach().numpy()
  
   # confusion matrix plotting
   cm = confusion_matrix(test_y.tolist(), pre)
   multi_label = ['P', 'S', 'N']
   tick_marks = np.arange(len(multi_label))
   plt.xticks(tick_marks, multi_label, rotation=45)
   plt.yticks(tick_marks, multi_label)
   thresh = cm.max() / 2
   for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
       plt.text(j, i, "{:,}".format(cm[i, j]),horizontalalignment="center",
           color="black" if cm[i, j] > thresh else "black",
           fontsize = 12)
   plt.ylabel('True label', fontsize = 12)
   plt.xlabel('Predicted label', fontsize = 12)
   plt.tight_layout()
   plt.imshow(cm, interpolation='nearest', cmap=plt.cm.GnBu)
   plt.title('Confusion Matrix', fontsize= 12)
   plt.colorbar()
   plt.savefig('./cm_resnet.jpeg', dpi =600)
   plt.close()

   print(metrics.classification_report(test_y.tolist(), pre, target_names = multi_label, digits=6))


if __name__ == "__main__":
    args = parser.parse_args() # Namespace object
    main(args)
