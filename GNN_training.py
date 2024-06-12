import pickle
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from GNN_model import MpGNN
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def train(train_loader):
    model.train()  # set the model to training mode
    for batch in train_loader:  # loop through the training set in a batch-wise fashion
        batch.to(device)  # move the batch to the device (GPU if applicable)
        optimizer.zero_grad()  # set gradients to 0
        out = model(batch)  # propagate the data through the model
        # print(out[:1,:], y_expanded[:1,:])
        loss = loss_func(out[:,0], batch.y[:,0]) + loss_func(out[:,1], batch.y[:,1])  # compute the loss
        loss.backward()  # derive gradients
        optimizer.step()  # update all parameters based on the gradients


def test(loader):
    model.eval()  # set the model to evaluation mode (no dropout)
    
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:  # loop through the supplied dataset in a batch-wise fashion
            data.to(device)  # transfer batch to device
            out = model(data)  # propagate the data through the model
            all_preds.append(out.cpu())
            all_labels.append(data.y.cpu())
    
    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    preds = (all_preds > 0.5).float()  # Binarize the predictions
    correct = (preds == all_labels).sum().item()
    total = all_labels.numel()
    
    return correct / total, all_labels, preds

def plot_train(train_accs, val_accs):
    fig, ax = plt.subplots(figsize=(8,6))
    fnt=16
    ax.plot(train_accs, color='blue', label='Train')
    ax.plot(val_accs, color='red', linestyle='--', label='Validation')
    ax.legend(fontsize=fnt)
    ax.tick_params(axis='both', labelsize=fnt)
    ax.set_xlabel('Epoch', fontsize=fnt)
    ax.set_ylabel('Accuracy', fontsize=fnt)

def plot_confusion_matrix(cm, class_name):
    fig, ax = plt.subplots(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title(f'Confusion Matrix for {class_name}')
    plt.show()

## Load data
with open(r'Data\Graph_dataset.pkl', 'rb') as f:
    molecules = pickle.load(f)

epochs = 10
batch_size = 32
node_feat_dim = 79
hidden_dim = 8
num_layers = 2
num_classes_per_classifier = 1   # for binary classifiers, num_classes_per_classifier is 1


train_data, val_data = train_test_split(molecules, test_size=0.05, random_state=45)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cpu'
print(f'Loaded device: {device}')

model = MpGNN(num_features=node_feat_dim, hidden_channels=hidden_dim, num_layers=num_layers, num_classes=2) # initialize our GNN
# print(model)
model.to(device)  # and move to the GPU, if possible

optimizer = torch.optim.Adam(model.parameters())  # adam usually works well
loss_func = torch.nn.BCELoss() # binary classification, so crossentropy is a suitable loss

train_accs = []
val_accs = []
start = time.time()
for epoch in range(1, epochs):  
    train(train_loader)  # do one training step over the entire dataset
    with torch.no_grad():
        train_acc, train_labels, train_preds = test(train_loader)  # compute the training accuracy
        val_acc, val_labels, val_preds = test(val_loader)  # compute the validation accuracy
    tic = time.time()
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}. Training time so far: {tic-start:.1f} s')
    train_accs.append(train_acc)  # save accuracies so we can plot them
    val_accs.append(val_acc)

plot_train(train_accs, val_accs)
plt.show()