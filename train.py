"""
trains an evaluates the last layer of the model to recognise the crystal systems
"""

import torch 
from torch import optim
import torch.nn as nn 
import torchvision.models as models 
from dataset import loader_train, loader_test, dataset 
import joblib
import os

#perapring the model 
resnet18 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT) #loading the model with pretrained layers


for param in resnet18.parameters(): 
    param.requires_grad = False #freezing all layers


num_target_classes = 7 
in_features = resnet18.fc.in_features

resnet18.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(in_features, num_target_classes)
) #replacing the final layer 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #gpu support for training
resnet18 = resnet18.to(device)
print(f"Using device: {device}")
criterion = nn.CrossEntropyLoss() #calculaets how wrong the logit score is 
optimizer = optim.Adam(
    filter(lambda p: p.requires_grad, resnet18.parameters()),
    lr=0.0001, 
    weight_decay=1e-4 #penalize overly complex solutions
)


#training loop 
num_epoches = 12

for epoche in range (num_epoches): #loop for the epochs
    resnet18.train()
    running_loss = 0.0 #loss accumulator 

    for images, labels in loader_train: #loop for each batch in a data loader 
        images, labels = images.to(device), labels.to(device) #gpu 
        optimizer.zero_grad() 

        outputs = resnet18(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() #accumulating the average loss of each batch 

    epoch_loss = running_loss/len(loader_train) #average loss 
    print(f"epoch number: {epoche+1}/{num_epoches} - loss: {epoch_loss:.4f}")


#evaluation 
resnet18.eval()
total_correct =0 
total_images = 0
with torch.no_grad(): 
    for images, labels in loader_test: 
        images, labels = images.to(device), labels.to(device) #gpu
        outputs = resnet18(images)
        loss = criterion(outputs, labels)

        predicts = torch.argmax(outputs, dim=1) #predicted classes with the most percentage for each image
        correct_preds = (predicts==labels) #comparing trained model's predictions and true labels 

        total_correct += correct_preds.sum().item() #accumulating the correct answers 
        total_images += labels.size(0)
accuracy = total_correct/total_images
print(f"Test accuracy: {accuracy: .2%}")

os.makedirs("model", exist_ok=True)

# save the model
torch.save(resnet18.state_dict(), "model/model.pth")

# save the class names (needed by app.py)
joblib.dump(dataset.classes, "model/classes.pkl")

print("Model saved!")

# Diagnostic check — see what the model actually predicts
from collections import Counter

resnet18.eval()
all_preds = []
with torch.no_grad():
    for images, labels in loader_test:
        images = images.to(device)
        outputs = resnet18(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())

print("Prediction distribution:", Counter(all_preds))
print("Class mapping:", {i: name for i, name in enumerate(dataset.classes)})


