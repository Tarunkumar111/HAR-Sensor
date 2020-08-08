import time
import random
import sys
import numpy as np
import torch
import argparse
import torch.nn.functional as F
import matplotlib.pyplot as plt
from statistics import mean, pstdev
import torch.nn.functional as F

from model import *
from dataloader import *

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('cuda', action='store_true', default=False,
                    help='Use CUDA for training.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=200,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.01,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay for optimizer.')
parser.add_argument('--hidden', type=int, default=16,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--train_split', type=float, default=0.8,
                    help='Ratio of train split from entire dataset.Rest goes to test set')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch size for loading mini batches of data')
parser.add_argument('--dataset_name', type=str, default='WISDM',
                    help='Dataset name')

args = parser.parse_args()
if args.cuda:
	if torch.cuda.is_available():
		device = torch.device('cuda')
		torch.cuda.manual_seed(args.seed)
	else:
		print("Sorry no gpu found!!")
		device=torch.device('cpu')
		print("Running model on cpu")
else:
	device=torch.device('cpu')

#Setting seed to reproduce results
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = GraphDataset(root='/content/drive/My Drive/GraphTrain/dataset/', name=args.dataset_name, use_node_attr=True)
data_size = len(dataset)
# print("*"*10)
# print(dataset)
# print(dataset[0])
# print(dataset[1])
# print(data_size)
# print(dataset.num_features)
# print(args.hidden)
# print(args.dropout)
# print(dataset.num_classes)
# print("*"*10)

#train
def train():
    model.train()

    loss_all = 0
    train_correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        train_correct +=output.max(dim=1)[1].eq(data.y).sum().item()
        optimizer.step()
    return loss_all / len(train_loader.dataset), train_correct/len(train_loader.dataset)

#test
def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)

# Main code for training
if __name__ == "__main__":

    print("*"*30)
    print("Performing experiment 5 times")
    print("*"*30)
    #for performing experiment 5 times and finding the best result with random shuffled dataset each time
    for i in range(5):
        #for shuffling the graphs
        dataset = dataset.shuffle()  
        #Spliting train and test set based on train_Split ratio
        train_dataset = dataset[:int(data_size * args.train_split)]
        test_dataset = dataset[int(data_size * args.train_split):]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

        #Model and Optimizer
        model = Net(dataset.num_features,args.hidden, args.dropout,dataset.num_classes).to(device)
        #Change optimizer as needed(Eg SGD)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        loss_values = []
        accuracy_values = []
        train_start = time.time()
        for epoch in range(args.epochs):   
            loss, train_acc = train()
            loss_values.append(loss)
            accuracy_values.append(train_acc)
            #train_acc = test(train_loader)
            # print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f}'.
            #       format(epoch, loss, train_acc, test_acc))
            print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}'.format(epoch, loss, train_acc))
        train_end = time.time()
        print("Training time : ", train_end - train_start)

        fig = plt.figure(figsize=(5,5))
        ax = fig.add_subplot(111)
        ax.set_title('Training loss and accuracy Vs Epoch')
        plt.plot(loss_values, color='red',label='Loss')
        plt.plot(accuracy_values, color='blue',label='Accuracy')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss and Accuracy')
        ax.legend(loc='best')
        #plt.show()
        plt.savefig(f"5times_{i+1}.png")

        test_start = time.time()
        test_acc = test(test_loader)
        test_end = time.time()
        print("Testing time : ", test_end - test_start)
        print("Test accuracy : ", test_acc)

    print()
    print()
    print("*"*30)
    print("iterative progressive experiment")
    print("*"*30)
    #for iterative training experiment
    ratios=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    train_values = []
    test_values = []
    for r in ratios:
        dataset = dataset.shuffle()  
        train_dataset = dataset[:int(data_size * r)]
        test_dataset = dataset[int(data_size * r):]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        print("No of train graph", len(train_loader.dataset))
        print("No of test graph", len(test_loader.dataset))
        model = Net(dataset.num_features,args.hidden, args.dropout,dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        for epoch in range(args.epochs):
            loss, train_acc = train()
            #print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}'.format(epoch, loss, train_acc))
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        train_values.append(train_acc)
        test_values.append(test_acc)
        print("Train accuracy : ", train_acc)
        print("Test accuracy : ", test_acc)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111)
    ax.set_title('Training and testing accuracy Vs Train Ratio')
    plt.plot(ratios, train_values, color='red', marker= 'o',label='Train Acc')
    plt.plot(ratios, test_values, color='blue',marker= 'o', label='Test Acc')
    ax.set_xlabel('Train Ratio')
    ax.set_ylabel('Training and testing accuracy')
    ax.legend(loc='best')
    #plt.show()
    plt.savefig(f"iterative.png")

    print()
    print()
    print("*"*30)
    print("Getting mean and standard deviation")
    print("*"*30)
    #for finding mean accuracy
    testAccus=[]
    for i in range(5):
        dataset = dataset.shuffle()  
        train_dataset = dataset[:int(data_size * args.train_split)]
        test_dataset = dataset[int(data_size * args.train_split):]
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
        print("No of train graph", len(train_loader.dataset))
        print("No of test graph", len(test_loader.dataset))
        model = Net(dataset.num_features,args.hidden, args.dropout,dataset.num_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)    
        for epoch in range(args.epochs):
            loss, train_acc = train()
            #print('Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}'.format(epoch, loss, train_acc))
        test_acc = test(test_loader)
        print("Test accuracy : ", test_acc)
        testAccus.append(test_acc)

    m=mean(testAccus)
    print("mean: ",m)
    sd=pstdev(testAccus)
    print("sd: ", sd)







