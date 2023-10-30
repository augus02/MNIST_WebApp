from __future__ import annotations

from itertools import cycle
import torch.onnx
from torch.optim import Optimizer, SGD
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import (DataLoader, Dataset)
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T


class MNISTDataset(Dataset):
    def __init__(self, train: bool, path: str, device: torch.device) -> None:
        super().__init__()
        self.prefix = 'train' if train else 'test'
        self.path = path

        self.path_xs = os.path.join(self.path, f'mnist_{self.prefix}_xs.pt')
        self.path_ys = os.path.join(self.path, f'mnist_{self.prefix}_ys.pt')

        

        if not os.path.exists(self.path_xs) or not os.path.exists(self.path_ys):
            set = MNIST(path, train=train, download=False, transform=T.ToTensor())
            loader = DataLoader(set, batch_size=batch_size, shuffle=train)
            n = len(set)

            xs = torch.empty((n, *set[0][0].shape), dtype=torch.float32)
            ys = torch.empty((n, ), dtype=torch.int64)

            for i,(x, y) in tqdm(loader, desc=f'Preparing {self.prefix.capitalize()} Set'):
                xs[i] = x
                ys[i] = y
                
            torch.save(xs, self.path_xs)
            torch.save(ys, self.path_ys)


        self.device = device
        self.xs = torch.load(self.path_xs, map_location=self.device)
        self.ys = torch.load(self.path_ys, map_location=self.device)

    def __len__(self) -> int:
        return self.xs.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.xs[idx], self.ys[idx]
    

class FeaturesDetector(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels = 24, out_channels=out_channels, kernel_size=3, stride = 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(torch.max_pool2d(self.conv1(x), 2))
        x = torch.relu(torch.max_pool2d(self.conv2(x), 2))
        return x
    
class MNISTClassifier(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.softmax(self.fc3(x), dim=1)
        return x
    
class CNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = FeaturesDetector(1, 32)
        self.classifier = MNISTClassifier(800, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(x.shape)
        # x = x.reshape(280, 280, 4)
        # x = x[:, :, 3]
        # x = x.reshape(1, 1, 280, 280)
        # x = F.avg_pool2d(x, 10)
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
    
    def fit(self, train_loader: DataLoader, optimizer: Optimizer, scheduler, epochs: int) -> None:
        best_loss = 1e20
        for epoch in range(epochs):
            self.train()
            for x, y in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                optimizer.zero_grad()
                y_hat = self(x)
                loss = F.nll_loss(y_hat, y)
                loss.backward()
                if loss.item() < best_loss:
                    torch.save(self.state_dict(), 'mnist_cnn.pt')
                    best_loss = loss.item()
                optimizer.step()
                scheduler.step()

            self.eval()
            correct = 0
            for x, y in tqdm(test_loader, desc=f'Epoch {epoch + 1}/{epochs}'):
                y_hat = self(x)
                loss += F.nll_loss(y_hat, y, reduction='sum').item()
                correct += (torch.argmax(y_hat, dim=1) == y).float().sum().item()

            print()
            print(f'Loss: {loss / len(test_loader.dataset):.2e}')
            print(f'Accuracy: {correct / len(test_loader.dataset)}')
            print()

    @torch.inference_mode()
    def test(self, loader: DataLoader) -> None:
        self.eval()
        correct = 0
        loss = 0
        for x, y in tqdm(loader, total=len(loader),desc='Testing'):
            y_hat = self(x)
            loss += F.nll_loss(y_hat, y, reduction='sum').item()
            correct += (torch.argmax(y_hat, dim=1) == y).float().sum().item()
        print()
        print(f'Loss: {loss / len(loader.dataset):.2e}')
        print(f'Accuracy: {correct / len(loader.dataset)}')
        print()


if __name__ == '__main__':
    device = torch.device('cpu')
    batch_size = 256
    epochs = 15
    lr = 1e-2

    train_set = MNISTDataset(train=True, path='tmp/datasets', device=device)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)

    test_set = MNISTDataset(train=False, path='tmp/datasets', device=device)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    model = CNN().to(device)

    if os.path.exists('mnist_cnn.pt'):
        model.load_state_dict(torch.load('mnist_cnn.pt'))

    optimizer = SGD(model.parameters(), lr=lr)
    scheduler = OneCycleLR(optimizer, lr, epochs=epochs, steps_per_epoch=len(train_loader))

    model.fit(train_loader, optimizer, scheduler, epochs)
    model.test(test_loader)

    torch.onnx.export(model, torch.randn(1, 1, 28, 28), 'onnx_mnist.onnx', verbose = True, opset_version = 9, input_names=["input"], output_names=["output"])
