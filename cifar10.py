import numpy as np
import torchvision
from torch.utils.data import Dataset

class CIFAR10(Dataset):
    
    def __init__(self, root = ".", train = True, download=False, transform=None, num_samples = None):
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        self.num_classes = 10          

        self.__dataset = torchvision.datasets.CIFAR10(root=root, 
                                                      train=train, 
                                                      download=download,
                                                      transform=transform)

        if num_samples is not None:
            self.__num_samples = min(num_samples, len(self.__dataset))
        else:
            self.__num_samples = len(self.__dataset)
                               
    def __len__(self):
        return self.__num_samples
    
    def __getitem__(self, idx):            
        return self.__dataset[idx]