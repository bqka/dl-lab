import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from itertools import product
import copy

transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset1 = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)

testset1 = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)

trainloader1 = torch.utils.data.DataLoader(trainset1, batch_size=64, shuffle=True)
testloader1 = torch.utils.data.DataLoader(testset1, batch_size=64)

trainset2 = torchvision.datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform)

testset2 = torchvision.datasets.CIFAR100(root='./data', train=False,
                                          download=True, transform=transform)

trainloader2 = torch.utils.data.DataLoader(trainset2, batch_size=64, shuffle=True)
testloader2 = torch.utils.data.DataLoader(testset2, batch_size=64)

class CNN(nn.Module):
    def __init__(self, act_fn, num_classes=10):
        super().__init__()

        self.act = act_fn

        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),
            nn.BatchNorm2d(32),
            act_fn,
            nn.MaxPool2d(2),

            nn.Conv2d(32,64,3,padding=1),
            nn.BatchNorm2d(64),
            act_fn,
            nn.MaxPool2d(2),

            nn.Conv2d(64,128,3,padding=1),
            nn.BatchNorm2d(128),
            act_fn,
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*4*4,256),
            act_fn,
            nn.Dropout(0.5),
            nn.Linear(256,num_classes)
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def init_weights(model, method):
    for m in model.modules():
        if isinstance(m,(nn.Conv2d,nn.Linear)):
            if method=="xavier":
                nn.init.xavier_uniform_(m.weight)
            elif method=="kaiming":
                nn.init.kaiming_uniform_(m.weight)
            elif method=="random":
                nn.init.uniform_(m.weight)

def train_model(model,trainloader,testloader,optimizer,epochs=5):
    criterion = nn.CrossEntropyLoss()
    best_acc = 0
    best_weights = None

    for epoch in range(epochs):
        model.train()
        for imgs,labels in trainloader:
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out,labels)
            loss.backward()
            optimizer.step()

        acc = evaluate(model,testloader)

        if acc>best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(model.state_dict())

        print(f"Epoch {epoch} Acc {acc:.2f}")

    return best_acc,best_weights

def evaluate(model,loader):
    model.eval()
    correct,total=0,0

    with torch.no_grad():
        for imgs,labels in loader:
            out = model(imgs)
            _,pred = torch.max(out,1)
            total+=labels.size(0)
            correct+=(pred==labels).sum().item()

    return 100*correct/total

activations={
    "relu":nn.ReLU(),
    "tanh":nn.Tanh(),
    "leaky":nn.LeakyReLU()
}

inits=["xavier","kaiming","random"]
opts=["sgd","adam","rmsprop"]

results={}
best_global=0
best_model=None

for act_name,init,opt in product(activations.keys(),inits,opts):

    print(f"\nRunning {act_name}-{init}-{opt}")

    model=CNN(activations[act_name],10)

    init_weights(model,init)

    if opt=="sgd":
        optimizer=optim.SGD(model.parameters(),lr=0.01)
    elif opt=="adam":
        optimizer=optim.Adam(model.parameters(),lr=0.001)
    else:
        optimizer=optim.RMSprop(model.parameters(),lr=0.001)

    acc,weights=train_model(model,trainloader1,testloader1,optimizer)

    results[(act_name,init,opt)]=acc

    if acc>best_global:
        best_global=acc
        best_model=weights

torch.save(best_model,"best_cnn.pth")
