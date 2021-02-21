# Importing necessary libraries.

import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import os, random
from PIL import Image
import argparse
from extra import save_checkpoint, load_checkpoint
from torchvision.datasets import ImageFolder


def parse_args():
    
    # For the coding parser function I referred to the python documentation 
    # in this link  "https://docs.python.org/3/library/argparse.html".
    # Using argparse module will help the user to input values through the command line during runtime.
    # It is not necessary to see the program while using it, as arguments were added for important values.
    
    par = argparse.ArgumentParser(description="Training process")
    par.add_argument('--data_dir', action='store')
    par.add_argument('--arch', dest='arch', default='vgg16', choices=['vgg16', 'vgg19'])
    par.add_argument('--hidden_units', dest='hidden_units', default='512')
    par.add_argument('--epochs', dest='epochs', default='1')
    par.add_argument('--learning_rate', dest='learning_rate', default='0.0005')
    par.add_argument('--gpu', action="store_true", default=True)
    return par.parse_args()

def main():
    
    args = parse_args()
    
    # For the code below I referred to the part8 Transfer learning solution in this module.
    # Directories assignment.
    
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Following transforms from the torchvision are used to modify the image. The rotation method helps to rotate the image,
    # crop method for cropping, resize method to resize the image, flip method for flipping the image, and 
    # normalize method to set the standard deviation and mean values.
   
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    test_transforms = transforms.Compose([transforms.Resize(224),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    
    # The image_datasets take images from the directory and apply the above training,validation and testing transform.
    
    image_datasets = [ImageFolder(train_dir, transform=train_transforms),
                      ImageFolder(valid_dir, transform=valid_transforms),
                      ImageFolder(test_dir, transform=test_transforms)]
    
    # The dataloaders take data from image_datasets in the batch size of 64 and shuffles the remaining data.
    
    dataloaders = [torch.utils.data.DataLoader(image_datasets[0], batch_size=64, shuffle=True),
                   torch.utils.data.DataLoader(image_datasets[1], batch_size=64),
                   torch.utils.data.DataLoader(image_datasets[2], batch_size=64)]
    
    # User can select the model through argparse from two options, the default model is vgg16.
    
    model = getattr(models, args.arch)(pretrained=True)
    
    # Changes the device to GPU if it is available otherwise it will run in CPU mode.
    
    gorc = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    
    # To freeze the parameters.
    
    for param in model.parameters():
        param.requires_grad = False
    
    # The first line of code applies a linear transformation and passes the output to ReLU activation. 
    # After applying dropout to prevent overfitting, output passed as input to the linear transformation.
    # The last line will apply the softmax function to the output unit.
    
    model.classifier = nn.Sequential(nn.Linear(25088, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1))
    
    # If and elif statements help the user to choose the model provided in the argparse.
    
    if args.arch == "vgg16":
        
        feature_num = model.classifier[0].in_features
        model.classifier = nn.Sequential(nn.Linear(feature_num, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1))

    elif args.arch == "vgg19":
        
        feature_num = model.classifier[0].in_features
        model.classifier = nn.Sequential(nn.Linear(feature_num, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.5),
                                 nn.Linear(1024, 102),
                                 nn.LogSoftmax(dim=1))
    
    # Below statements call functions save_checkpoint function, argparse function, and optimizer (used
    # to update wieghts).
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=float(args.learning_rate))
    model.to(gorc);
    epochs = int(args.epochs)
    class_index = image_datasets[0].class_to_idx
    gpu = args.gpu
    train(model, criterion, optimizer, dataloaders, epochs,gpu)
    model.class_to_idx = class_index
    save_checkpoint(model, optimizer, args, model.classifier)

def train(model, criterion, optimizer, dataloaders, epochs,gpu):
    
    # For the code below I referred to the part8 Transfer learning solution in this module. 
    # Changes the device to GPU if it is available otherwise it will run in CPU mode.
    
    gorc = torch.device("cuda" if torch.cuda.is_available() else "cpu")     
    
    # Intializing values
    
    print_every = 5
    run_loss = 0
    step = 0
    
    for epoch in range(epochs):
        
        # Code for training data.
        
        for inputs, labels in dataloaders[0]:
            
            step += 1
            
            # Setting for the available device.
            
            inputs, labels = inputs.to(gorc), labels.to(gorc)
            
            # Multiple backward passes with the same parameters cause gradients to accumulate. 
            # Setting the optimizer to zero_grad to prevent gradients from previous training batches.
            
            optimizer.zero_grad()
            
            # Forward pass backward pass and updating the weights.
            
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
            run_loss += loss.item()
            
            if step % print_every == 0:
               
                valid_loss = 0
                accuracy = 0
                
                # Setting the dropout probability to zero for validation.
                
                model.eval()
                
                # Turning off gradients.
                
                with torch.no_grad():
                    
                    # Code for validation data.
                    
                    for inputs, labels in dataloaders[1]:
                        
                        inputs, labels = inputs.to(gorc), labels.to(gorc)
                        logps = model(inputs)
                        batch_loss = criterion(logps, labels)
                        valid_loss += batch_loss.item()

                        # Calculate accuracy
                        # Applying exponential to the probabilities.
                        
                        ps = torch.exp(logps)
                        
                        # Default value for topk is 5, which will help to display top 5 probabilities and indices.
                        
                        top_p, top_class = ps.topk(1, dim=1)
                        
                        # By using equals checking whether labels and top_class are matching. Applied (.view) 
                        # to labels to get the same shape as top_class.
                        
                        equals = top_class == labels.view(*top_class.shape)
                        
                        # Used mean to calculate the accuracy for the equals, here the equals is ByteTensor but the mean
                        # is applied on floattensor so converting it into floattensor.
                        # torch. mean returns scalar-tensor, to get the float values used the item().
                        
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                        
                # Printing epoch, training loss, testing loss, and test accuracy.
                
                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Train loss: {run_loss/print_every:.3f}.. "
                      f"Valid loss: {valid_loss/len(dataloaders[1]):.3f}.. "
                      f"valid accuracy: {accuracy/len(dataloaders[1]):.3f}")
                
                # Resetting running loss to zero and changing to training mode.
                
                run_loss = 0
                model.train()
    
    test_loss = 0
    accuracy = 0
    
    # Setting the dropout probability to zero for testing.
    
    model.eval()
    
    # Turning off gradients.
   
    with torch.no_grad():
        
        # Code for testing data.
        
        for inputs, labels in dataloaders[2]:
            
            inputs, labels = inputs.to(gorc), labels.to(gorc)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()
                    
            # Calculate accuracy
            # Applying exponential to the probabilities.
           
            ps = torch.exp(logps)
            
            # Default value for topk is 5, which will help to display top 5 probabilities and indices.
            
            top_p, top_class = ps.topk(1, dim=1)
            
            # By using equals checking whether labels and top_class are matching. Applied (.view) 
            # to labels to get the same shape as top_class.
           
            equals = top_class == labels.view(*top_class.shape)
            
            # Used mean to calculate the accuracy for the equals, here the equals is ByteTensor but the mean
            # is applied on floattensor so converting it into floattensor.
            # torch. mean returns scalar-tensor, to get the float values used the item().
            
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
        
        # Printing testing loss, and test accuracy.
        
        print(f"Test loss: {test_loss/len(dataloaders[2]):.3f}.. "
              f"Test accuracy: {accuracy/len(dataloaders[2]):.3f}")

if __name__ == "__main__":
    main()