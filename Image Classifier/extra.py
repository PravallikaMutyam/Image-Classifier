# Importing necessary libraries.

import torch
from torchvision import transforms, datasets
import json
import copy
import os
import argparse

# Parameters are saved in state_dict. For saving used torch.save and saved as checkpoint.pth.
# To load used torch.load. In checkpoint added epochs, model learning rate, and other important values. 
# Training the network every time is a difficult process, so by saving we can load from the checkpoint later.

def save_checkpoint(model, optimizer, args, classifier):
    
    # For the save checkpoint and load checkpoint I referred to the following links.
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html.
    # https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html.
    # https://stackoverflow.com/questions/42703500/best-way-to-save-a-trained-model-in-pytorch.
    # For the code below I referred to the Part 6 - Saving and Loading Models in this module.
    
    checkpoint = {'arch': args.arch, 
                  'model': model,
                  'hidden_units': args.hidden_units,
                  'model.classifier' : model.classifier,
                  'epochs': args.epochs,
                  'learning_rate': args.learning_rate,
                  'optimizer': optimizer.state_dict(),
                  'state_dict': model.state_dict(),
                  'class_to_idx': model.class_to_idx}

    torch.save(checkpoint, 'checkpoint.pth')
    
# Loaded checkpoint from the above-mentioned parameters.

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = checkpoint['model']
    model.classifier = checkpoint['model.classifier']
    epochs = checkpoint['epochs']
    learning_rate = checkpoint['learning_rate']
    optimizer = checkpoint['optimizer']
    
    # below line loads the state dict into the network.
    
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def load_cat_names(filename):
    with open(filename) as f:
        category_names = json.load(f)
    return category_names