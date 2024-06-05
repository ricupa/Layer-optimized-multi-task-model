import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleTaskModel(nn.Module):
    """ Single-task baseline model with encoder + decoder """
    def __init__(self, backbone: nn.Module, decoders: nn.Module, task: str):
        super(SingleTaskModel, self).__init__()
        self.backbone = backbone
        self.decoders = decoders 
        self.task = task

    def forward(self, x):
        # out_size = x.size()[2:]
        out = self.decoders(self.backbone(x))
        return {self.task: out}


class MultiTaskModel(nn.Module):
   
    def __init__(self, backbone: nn.Module, decoders: nn.ModuleDict, tasks: list):
        super(MultiTaskModel, self).__init__()
        assert(set(decoders.keys()) == set(tasks))
        self.backbone = backbone
        self.decoders = decoders
        self.tasks = tasks

    def forward(self, x):
        return {task: self.decoders[task](self.backbone(x)) for task in self.tasks}
        
        # shared_representation = self.backbone(x)
        # shared_representation.requires_grad_()
        # print(shared_representation.shape)
        # out = {}
        # for task in self.tasks:
        #     out[task] = self.decoders[task](shared_representation)            
        # return out
        
        
    
    
    
class Lambda_MLP(nn.Module):
    def __init__(self, in_features):
        super(Lambda_MLP, self).__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(in_features, 128)  # First hidden layer with 128 neurons
        self.fc2 = nn.Linear(128, 64)  # Second hidden layer with 64 neurons
        self.fc3 = nn.Linear(64, 1)  # Output layer with 1 neuron
        
        # Define ReLU activation function and a dropout layer
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        
    def forward(self, x):
        # Define the forward pass
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        return x
    
    
    



