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
from extra import load_checkpoint, load_cat_names
import tensorflow as tf
import pandas as pd


def parse_args():
    
    # For the coding parser function I referred to the python documentation in this link  
    # "https://docs.python.org/3/library/argparse.html".
    # Using argparse module will help the user to input values through the command line during runtime.
    # It is not necessary to see the program while using it, as arguments were added for important values
    # like filepath, top_k, etc...
    
    pre = argparse.ArgumentParser()
    pre.add_argument('--checkpoint', action='store', default='checkpoint.pth')
    pre.add_argument('--top_k', dest='top_k', default='5')
    pre.add_argument('--filepath', dest='filepath', default=None)
    pre.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    pre.add_argument('--gpu', action='store_true', default=True)
    return pre.parse_args()

def process_image(image):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # For pil, thumbnail, and transpose implementation referred documentation reference given in this module.
    # For crop I went through the link "https://discuss.pytorch.org/t/how-to-crop-image-tensor-in-model/8409/2".
    # from pil Image open the required image.
    
    fimg = Image.open(image)
    
    # thumbnail function used to set the size of the image.
    
    fimg.thumbnail((256,256))
    
    # Crop method to crop the image.
    
    fimg = fimg.crop((1,1,255,255))
   
    # To set the values between 0 and 1.
    
    fimg = np.array(fimg)/255
    
    # Set standard deviation to required values.
    
    stnd = np.array([0.229, 0.224, 0.225])
    
    # Set mean to required values.
    
    mean = np.array([0.485, 0.456, 0.406])
    fimg = (fimg - mean) / stnd
    
    # In PyTorch color channel is the first dimension so by using the transpose method changing the
    # color dimension to the first dimension.
    
    fimg = fimg.transpose(2,0,1)
    
    # Returning fimg
    
    return fimg

def predict(imgpath, model, topk, gpu):
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    # Changes the device to GPU if it is available otherwise it will run in CPU mode.
    
    gorc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Taking the output from the process image as input.
    
    fimg = process_image(imgpath)
    
    # For the following code I referred to the section Part 1 - Tensors in PyTorch solution in this module and also 
    # I referred in this link "https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html".
    # Converting numpy image to torch tensor, and for converting to the float tensor used the float.
    
    fimg = np.array([fimg])
    fimg = torch.from_numpy(fimg).float()
    fimg = (fimg).to(gorc)
    
    # For the Code below I referred to part 5 Inference and validation (Solution)
    # Applying forward pass.
    
    result = model.forward(fimg)
    
    # Applying exponential to the result.
    
    ps = torch.exp(result)
    
    # Taking input from ps and apply topk on it, then converting it into a list and saving it in probs and classes.
    
    probs = torch.topk(ps, topk)[0].tolist()[0] 
    classes = torch.topk(ps, topk)[1].tolist()[0] 
    
    # Empty list is created i.e. list1.
    
    list1 = []
    
    x = len(model.class_to_idx.items())
    y = 0
    
    while y < x:
        
        # Append the value to the list1.
        
        list1.append(list(model.class_to_idx.items())[y][0])
        
        # The y value is incremented by 1
        
        y += 1
    
    # creates another empty list i.e. list2
    
    list2 = []
    
    # Assigning
    
    i = 0
    c = topk
    
    while i < c :
        
        # Classes value from the list1 is appended to list2.
        
        list2.append(list1[classes[i]])
        
        # i value is incremented by 1.
        
        i += 1
    
    # Returning probs and list2.
    
    return probs, list2
    
def main(): 
    
    # Take values from argparse and load to cat_to_name.
    
    args = parse_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_names(args.category_names)
    
    # If filepath is not provided the following code will execute.
    
    if args.filepath == None:
        
        # Randomly selects the number between 0 and 101 and assigns the value to img1.
        
        img1 = random.randint(1, 102)
        
        # Output of img1 is added, selects randomly image from the following directory, and assigns to the image.
        
        image = random.choice(os.listdir('./flowers/valid/' + str(img1) + '/'))
        
        # img1 and image values are added to 'flowers/valid' directory and assigned to image_path.
        
        img_path = './flowers/valid/' + str(img1) + '/' + image
        
        # Above value from image_path will pass as input to the predict function and executes the predict function.
        
        prob, classes = predict(img_path, model, int(args.top_k), gpu)
        
    else:
        
        # if the user gives filepath will store in img_path.
        
        img_path = args.filepath
        
        # Above user given value from image_path will pass as input to the predict function and executes
        # the predict function.
        
        prob, classes = predict(img_path, model, int(args.top_k), gpu)
    
    # Printing prob,classes and cat_to_name.   
    
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])
    
    # Changes the device to GPU if it is available otherwise it will run in CPU mode.
    
    gorc = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(gorc);
    
if __name__ == "__main__":
    main()