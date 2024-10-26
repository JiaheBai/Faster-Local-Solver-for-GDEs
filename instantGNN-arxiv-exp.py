import numpy as np
import scipy.sparse as sp
import time
import gc
import sys
from InstantGNN import *
from tqdm import tqdm as tqdm
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


class MyDataset(Dataset):
    def __init__(self, features, labels):
        super(MyDataset, self).__init__()

        self.label = torch.tensor(labels)
        self.data = torch.tensor(features.astype(np.float32))

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return self.data.size(0)


class ClassMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(ClassMLP, self).__init__()

        self.lins = nn.ModuleList()
        self.lins.append(nn.Linear(in_channels, hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for ii in range(num_layers - 2):
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def trainModel(model, trainloader, validloader, device, check_file, optimizer, epochs, patience, show_tqdm = False):
    #best_f1 = 0
    best_f1 = 0
    bad_count = 0
    traintime = 0
    if show_tqdm:
        for epoch in tqdm(range(epochs)):
            model.train()
            st = time.time()
            for i, (X_train, y_train) in enumerate(trainloader):
                X_train = X_train.to(device)
                y_train = y_train.to(device)
                pred = model(X_train)
                l = F.nll_loss(pred, y_train.flatten())

                optimizer.zero_grad()
                l.backward()
                optimizer.step()
            traintime += time.time() - st
            model.eval()
            estimates = []
            targets = []
            for X_test, Y_test in validloader:
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                outputs = model(X_test)
                _, predicted = torch.max(outputs.data, 1)
                estimates.extend(predicted.cpu().detach().numpy())  # extend可以添加多个元素到列表的末尾
                targets.extend(Y_test.cpu().detach().numpy())
            f1 = f1_score(targets, estimates, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                torch.save({'model': model.state_dict(), 'best_f1': best_f1}, check_file)
                bad_count = 0
            else:
                bad_count += 1
            if bad_count >= patience:
                break
    else:
        for epoch in range(epochs):
            model.train()
            st = time.time()
            for i, (X_train, y_train) in enumerate(trainloader):
                X_train = X_train.to(device)
                y_train = y_train.to(device)
                pred = model(X_train)
                l = F.nll_loss(pred, y_train.flatten())

                optimizer.zero_grad()
                l.backward()
                optimizer.step()

            traintime += time.time() - st
            model.eval()
            estimates = []
            targets = []
            for X_test, Y_test in validloader:
                X_test = X_test.to(device)
                Y_test = Y_test.to(device)
                outputs = model(X_test)
                _, predicted = torch.max(outputs.data, 1)
                estimates.extend(predicted.cpu().detach().numpy())  # extend可以添加多个元素到列表的末尾
                targets.extend(Y_test.cpu().detach().numpy())
            f1 = f1_score(targets, estimates, average='weighted')
            if f1 > best_f1:
                best_f1 = f1
                torch.save({'model': model.state_dict(), 'best_f1': best_f1}, check_file)
                bad_count = 0
            else:
                bad_count += 1
            if bad_count >= patience:
                break
    return traintime


def evaluateModel(model, trainloader, validloader, testloader, device):
    model.eval()
    total = 0
    correct = 0
    for X_test, Y_test in trainloader:
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total += Y_test.size(0)
        correct += (predicted.reshape(Y_test.shape) == Y_test).sum().item()
    train_acc = correct / total
    total = 0
    correct = 0
    for X_test, Y_test in validloader:
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total += Y_test.size(0)
        correct += (predicted.reshape(Y_test.shape) == Y_test).sum().item()
    valid_acc = correct / total
    total = 0
    correct = 0
    for X_test, Y_test in testloader:
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs.data, 1)
        total += Y_test.size(0)
        correct += (predicted.reshape(Y_test.shape) == Y_test).sum().item()
    test_acc = correct / total
    return train_acc, valid_acc, test_acc


dataset = np.load('dataset/ogbn-arxiv-exp.npz', allow_pickle=True)
indptr = dataset['indptr']
indices = dataset['indices'].astype(np.int32)
degree = dataset['degree']
Us = dataset['Us'].astype(np.int32)
Vs = dataset['Vs'].astype(np.int32)
node_feature = dataset['node_feature'].astype(np.float32)
labels = dataset['labels']
n = len(labels)
m = len(indices)
split_idx = dataset['split_idx']
split_idx = split_idx.reshape(1, -1)[0]
split_idx = split_idx[0]
train_idx = split_idx['train']
val_idx = split_idx['valid']
test_idx = split_idx['test']
scaler = StandardScaler()
#scaler.fit(node_feature[train_idx])
scaler.fit(node_feature)
node_feature = scaler.transform(node_feature)
del scaler
gc.collect()
gc.collect()
snapshots = len(Us)
eps = 0.01/n
alpha = 0.1
beta = 0.5
alpha = alpha / (2 - alpha)
mu = (1. - alpha) / (1. + alpha)
opt_omega = 1. + (mu / (1. + np.sqrt(1. - mu ** 2.))) ** 2.
alpha = 2 * alpha / (1 + alpha)
epoch_num = 1000
train_batch = 8192
test_batch = 1024
patience = 50
check_file = 'ogbn-arxiv_model.pth'
hidden_channels = 1024
num_layers = 4
dropout_rate = 0.3
learning_rate = 1e-4
weight_decay = 0
random_seed = 17

algo = 'fwdbatchinsert'
model = ClassMLP(in_channels=len(node_feature[0]), hidden_channels=hidden_channels, out_channels=172,
                 num_layers=num_layers, dropout=dropout_rate)
model.to(device)
times = np.zeros(snapshots + 1)
traintimes = np.zeros(snapshots + 1)
infertimes = np.zeros(snapshots + 1)
train_accs = np.zeros(snapshots + 1)
valid_accs = np.zeros(snapshots + 1)
test_accs = np.zeros(snapshots + 1)

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

r = node_feature.copy()
x = node_feature
p = np.zeros(r.shape, dtype=np.float64)
pretime = instantGNNParallel(n, indptr, indices, degree, alpha, eps, p, r, beta, batchsize=50)
times[0] = pretime.sum()

trainset = MyDataset(p[train_idx], labels[train_idx])
trainloader = DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=0,drop_last=False)
validset = MyDataset(p[val_idx], labels[val_idx])
validloader = DataLoader(validset, batch_size=test_batch, shuffle=False, num_workers=0, drop_last=False)
testset = MyDataset(p[test_idx], labels[test_idx])
testloader = DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=0, drop_last=False)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
traintime = trainModel(model, trainloader, validloader, device, check_file, optimizer, epoch_num, patience)
traintimes[0] = traintime
state_dict = torch.load(check_file)
model.load_state_dict(state_dict['model'])
st = time.time()
train_acc, valid_acc, test_acc = evaluateModel(model, trainloader, validloader, testloader, device)
infertimes[0] = time.time() - st
train_accs[0] = train_acc
valid_accs[0] = valid_acc
test_accs[0] = test_acc

for snapshot in tqdm(range(snapshots)):
    us = Us[snapshot]
    vs = Vs[snapshot]
    pretime = instantGNNUpdateBatchParallel(n, indptr, indices, degree, alpha, eps, opt_omega, p, r, x, us, vs, beta, algo, batchsize=50)
    times[snapshot + 1] = pretime.sum()

    trainset = MyDataset(p[train_idx], labels[train_idx])
    trainloader = DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=0, drop_last=False)
    validset = MyDataset(p[val_idx], labels[val_idx])
    validloader = DataLoader(validset, batch_size=8192, shuffle=False, num_workers=0, drop_last=False)
    testset = MyDataset(p[test_idx], labels[test_idx])
    testloader = DataLoader(testset, batch_size=8192, shuffle=False, num_workers=0, drop_last=False)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    traintime = trainModel(model, trainloader, validloader, device, check_file, optimizer, epoch_num, patience)
    traintimes[snapshot + 1] = traintime
    state_dict = torch.load(check_file)
    model.load_state_dict(state_dict['model'])
    st = time.time()
    train_acc, valid_acc, test_acc = evaluateModel(model, trainloader, validloader, testloader, device)
    infertimes[snapshot + 1] = time.time() - st
    train_accs[snapshot + 1] = train_acc
    valid_accs[snapshot + 1] = valid_acc
    test_accs[snapshot + 1] = test_acc

np.savez('./results/instantGNN_ogbn-arxiv_fwd_result.npz', times=times, traintimes=traintimes, infertimes=infertimes, train_accs=train_accs, valid_accs=valid_accs, test_accs=test_accs)

del model

dataset = np.load('dataset/ogbn-arxiv-exp.npz', allow_pickle=True)
indptr = dataset['indptr']
indices = dataset['indices'].astype(np.int32)
degree = dataset['degree']
Us = dataset['Us'].astype(np.int32)
Vs = dataset['Vs'].astype(np.int32)
node_feature = dataset['node_feature'].astype(np.float32)
algo = 'sorbatchinsert'
model = ClassMLP(in_channels=len(node_feature[0]), hidden_channels=hidden_channels, out_channels=172,
                 num_layers=num_layers, dropout=dropout_rate)
model.to(device)
times = np.zeros(snapshots + 1)
traintimes = np.zeros(snapshots + 1)
infertimes = np.zeros(snapshots + 1)
train_accs = np.zeros(snapshots + 1)
valid_accs = np.zeros(snapshots + 1)
test_accs = np.zeros(snapshots + 1)

np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)

r = node_feature.copy()
x = node_feature
p = np.zeros(r.shape, dtype=np.float64)
pretime = instantGNNSorParallel(n, indptr, indices, degree, alpha, eps, opt_omega, p, r, beta, batchsize=50)
times[0] = pretime.sum()

trainset = MyDataset(p[train_idx], labels[train_idx])
trainloader = DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=0,drop_last=False)
validset = MyDataset(p[val_idx], labels[val_idx])
validloader = DataLoader(validset, batch_size=test_batch, shuffle=False, num_workers=0, drop_last=False)
testset = MyDataset(p[test_idx], labels[test_idx])
testloader = DataLoader(testset, batch_size=test_batch, shuffle=False, num_workers=0, drop_last=False)

model.reset_parameters()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
traintime = trainModel(model, trainloader, validloader, device, check_file, optimizer, epoch_num, patience)
traintimes[0] = traintime
state_dict = torch.load(check_file)
model.load_state_dict(state_dict['model'])
st = time.time()
train_acc, valid_acc, test_acc = evaluateModel(model, trainloader, validloader, testloader, device)
infertimes[0] = time.time() - st
train_accs[0] = train_acc
valid_accs[0] = valid_acc
test_accs[0] = test_acc

for snapshot in tqdm(range(snapshots)):
    us = Us[snapshot]
    vs = Vs[snapshot]
    pretime = instantGNNUpdateBatchParallel(n, indptr, indices, degree, alpha, eps, opt_omega, p, r, x, us, vs, beta, algo, batchsize=50)
    times[snapshot + 1] = pretime.sum()

    trainset = MyDataset(p[train_idx], labels[train_idx])
    trainloader = DataLoader(trainset, batch_size=train_batch, shuffle=True, num_workers=0, drop_last=False)
    validset = MyDataset(p[val_idx], labels[val_idx])
    validloader = DataLoader(validset, batch_size=8192, shuffle=False, num_workers=0, drop_last=False)
    testset = MyDataset(p[test_idx], labels[test_idx])
    testloader = DataLoader(testset, batch_size=8192, shuffle=False, num_workers=0, drop_last=False)

    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    traintime = trainModel(model, trainloader, validloader, device, check_file, optimizer, epoch_num, patience)
    traintimes[snapshot + 1] = traintime
    state_dict = torch.load(check_file)
    model.load_state_dict(state_dict['model'])
    st = time.time()
    train_acc, valid_acc, test_acc = evaluateModel(model, trainloader, validloader, testloader, device)
    infertimes[snapshot + 1] = time.time() - st
    train_accs[snapshot + 1] = train_acc
    valid_accs[snapshot + 1] = valid_acc
    test_accs[snapshot + 1] = test_acc

np.savez('./results/instantGNN_ogbn-arxiv_sor_result.npz', times=times, traintimes=traintimes, infertimes=infertimes, train_accs=train_accs, valid_accs=valid_accs, test_accs=test_accs)
