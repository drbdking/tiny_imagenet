import json 
import argparse


import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim



class MyNet(nn.Module):
    def __init__(self, num_classes=10, dropout=0.5, depth=3, width_factor=8) -> None:
        super().__init__()
        # Starting layer
        features_modules = [
            nn.Conv2d(3, width_factor, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            ]
        
        # Controlled by depth
        for i in range(1, depth):
            features_modules.append(nn.Conv2d(i * width_factor, (i + 1) * width_factor, kernel_size=3, padding=1))
            features_modules.append(nn.BatchNorm2d((i + 1) * width_factor))
            features_modules.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*features_modules)

        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear((i + 1) * width_factor * 3 * 3, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Arguments
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--width_factor", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-2)

    args = parser.parse_args()

    transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net = MyNet(depth=args.depth, width_factor=args.width_factor).to(device)

    # Loss func and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)

    # Training
    train_acc_record = []
    val_acc_record = []
    train_loss_record = []
    val_loss_record = []
    
    for epoch in range(50):  # loop over the dataset multiple times
        # Train 
        train_loss = 0.0
        num_correct = 0
        net.train()
        for i, data in enumerate(trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            preds = torch.argmax(outputs.detach(), dim=1)
            num_correct += torch.sum(preds == labels).detach().cpu().numpy()

            train_loss += loss.item()
        print(f"Epoch {epoch}, train loss: {train_loss / i}")
        print(f"Epoch {epoch}, train acc: {num_correct / (i * batch_size)}")
        train_acc_record.append(num_correct / (i * batch_size))
        train_loss_record.append(train_loss / i)

        # Val
        val_loss = 0.0
        num_correct = 0
        net.eval()
        with torch.no_grad():
            for i, data in enumerate(testloader):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward + backward + optimize
                outputs = net(inputs)
                preds = torch.argmax(outputs, dim=1)
                num_correct += torch.sum(preds == labels).detach().cpu().numpy()
                loss = criterion(outputs, labels)

                val_loss += loss.item()
        print(f"Epoch {epoch}, val loss: {val_loss / i}")
        print(f"Epoch {epoch}, acc: {num_correct / (i * batch_size)}")
        val_acc_record.append(num_correct / (i * batch_size))
        val_loss_record.append(val_loss / i)


    # Save record
    with open(f"output/d_{args.depth}_w_{args.width_factor}.json", 'w') as fp:
        rec = {'num_params': sum(p.numel() for p in net.parameters() if p.requires_grad),
               'train_loss': train_loss_record, 
               'train_acc': train_acc_record, 
               'val_loss': val_loss_record, 
               'val_acc': val_acc_record}
        json.dump(rec, fp)
        
