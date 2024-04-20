import torch
import torch.nn as nn
import torch.optim as optim
import os

# Define a simple neural network model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def get_model():
    model = SimpleModel()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion

def save_model(model, path="model.pth"):
    torch.save(model.state_dict(), path)
    return path

def load_model(path="model.pth"):
    model = SimpleModel()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model
