from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import torchvision.models as models



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)   # 64, 32, 26, 26
        x = F.relu(x)
        x = self.conv2(x)  # 64, 64, 24, 24
        x = F.relu(x)
        x = F.max_pool2d(x, 2)  # 64, 64, 12, 12
        x = self.dropout1(x)
        x = torch.flatten(x, 1)  # 64, 9216    
        x = self.fc1(x)   # 64, 128
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)    # 64, 10
        output = F.log_softmax(x, dim=1)   # 64, 10
        return output



if __name__ == "__main__":
    # Training settings
    batch_size = 64
    test_batch_size = 1000
    epochs = 15
    lr = 1.0
    gamma = 0.7
    log_interval = 10
    use_cuda = True
    torch.manual_seed(1)
    device = torch.device("cuda")
    train_kwargs = {'batch_size': batch_size}
    test_kwargs = {'batch_size': test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                    'pin_memory': True,
                    'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)


    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)   # 60000
    dataset2 = datasets.MNIST('../data', train=False, transform=transform)    # 10000
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Model().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=1.0)
    sdl = StepLR(optimizer, step_size=1, gamma=0.7)


    for epoch in range(epochs):
        model.train()
        # train
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                print("epoch: {}, train_loss: {}".format(epoch, loss.item()))
    
        # test 
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1)  # 保持维度不变
                correct += pred.eq(target).sum().item()
        nums = (batch_idx + 1) * test_batch_size
        test_avg_loss = test_loss / nums
        print('epoch:{}, test_avg_loss: {}, accuracy: {}'.format(epoch, test_avg_loss, correct/nums))
            
        sdl.step()

    torch.save(model.state_dict(), 'mnist.pt')

