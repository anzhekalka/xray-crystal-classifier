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

my_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.ToTensor(), #convert PIL Image to PyTorch Tensor
    transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std  = [0.229, 0.224, 0.225] #official ResNet training values 
    )
]) #transform pipeline

dataset = ImageFolder("data/crystals/", transform=my_transform) #loading all images from the crystals folder + transforming them


generator = Generator().manual_seed(42) #ensure the random split is exactly the same after each run
train_set, test_set = random_split(dataset, [0.8, 0.2],  generator=generator) #splitting dataset into train and test data

loader_train = DataLoader(train_set, batch_size=32, shuffle=True) #seperate data in batches
loader_test = DataLoader(test_set, batch_size=32, shuffle=False)
