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


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # ensure single channel
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.449], std=[0.226])  # grayscale ImageNet stats (averaged)
]) #transform pipeline for the training dataset

full_dataset = ImageFolder("data/crystals/", transform=train_transform) # full data sets with an applied tranforms

generator = Generator().manual_seed(42) #ensure the random split is exactly the same after each run
train_set, test_set = random_split(full_dataset, [0.8, 0.2], generator=generator) #splitting dataset into train and test data 

loader_train = DataLoader(train_set, batch_size=32, shuffle=True) #seperate data in batches
loader_test = DataLoader(test_set, batch_size=32, shuffle=False)

class_names = full_dataset.classes
print(f"Class names: {class_names}")
print(f"Number of training images: {len(train_set)}")
print(f"Number of test images: {len(test_set)}")

dataset = full_dataset
