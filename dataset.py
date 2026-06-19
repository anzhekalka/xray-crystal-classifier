"""
Loads all images using ImageFolder, 
splits into 80% train / 20% test using random_split, 
Wraps both in DataLoader with batch_size=32
"""


from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data import random_split
from torch import Generator

evaluate_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), #convert PIL Image to PyTorch Tensor
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225] #official ResNet training values 
    )
]) #transform pipeline for the evaluation dataset

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
]) #transform pipeline for the training dataset

train_dataset = ImageFolder("data/crystals/", transform=train_transform) 
test_dataset = ImageFolder("data/crystals/", transform=evaluate_transform)#two full data sets with different tranforms

generator = Generator().manual_seed(42) #ensure the random split is exactly the same after each run
train_indices, test_indices = random_split(
    range(len(train_dataset)), [0.8, 0.2], generator=generator
 ) #splitting dataset into train and test data 

from torch.utils.data import Subset
train_set = Subset(train_dataset, train_indices.indices)
test_set  = Subset(test_dataset, test_indices.indices)

loader_train = DataLoader(train_set, batch_size=32, shuffle=True) #seperate data in batches
loader_test = DataLoader(test_set, batch_size=32, shuffle=False)

dataset = train_dataset
