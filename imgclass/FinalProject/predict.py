import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
import numpy as np

from PIL import Image
from train import load_arch

import json
import argparse

# Define command line arguments
arguments_parser = argparse.ArgumentParser(description='Predictive network')
arguments_parser.add_argument('--image', action="store",type=str, help='Image to predict')
arguments_parser.add_argument('--checkpoint', action="store",type=str, help='Model checkpoint to use when predicting')
arguments_parser.add_argument('--topk', type=int, action="store",help='Return top K predictions')
arguments_parser.add_argument('--labels', action="store",type=str, help='JSON file containing label names')
arguments_parser.add_argument('--gpu', action='store_true', help='Use GPU if available')

args, _ = arguments_parser.parse_known_args()


# Implement the code to predict the class from an image file
def predict(inputs, checkpoint, topk=5, labels='', gpu=False):
    ''' Predict deep learning model.
    '''
    # Use command line values when specified
    if args.inputs:
        image = args.inputs     
        
    if args.checkpoint:
        checkpoint = args.checkpoint

    if args.topk:
        topk = args.topk
            
    if args.labels:
        labels = args.labels

    if args.gpu:
        gpu = args.gpu
    
    # Load the checkpoint
    checkpoint_dict = torch.load(checkpoint)

    arch = checkpoint_dict['arch']
    num_labels = len(checkpoint_dict['class_to_idx'])
    hidden_units = checkpoint_dict['hidden_units']
        
    model = load_model(arch=arch, num_labels=num_labels, hidden_units=hidden_units)

    # Apply GPU CUDA
    if gpu and torch.cuda.is_available():
        model.cuda()
        
    was_training = model.training    
    model.eval()
    
    img_loader = transforms.Compose([
        transforms.Resize(256), 
        transforms.CenterCrop(224), 
        transforms.ToTensor()])
    
    pil_image = Image.open(inputs)
    pil_image = img_loader(pil_image).float()
    
    inputs = np.array(pil_image)    
    

    inputs = (np.transpose(inputs, (1, 2, 0)) - np.array([0.485, 0.456, 0.406]))/np.array([0.229, 0.224, 0.225])    
    inputs = np.transpose(inputs, (2, 0, 1))
    
    inputs = Variable(torch.FloatTensor(inputs), requires_grad=True)
    inputs = inputs.unsqueeze(0) # this is for VGG
    
    if gpu and torch.cuda.is_available():
        inputs = inputs.cuda()
            
    result = model(inputs).topk(topk)

    if gpu and torch.cuda.is_available():

        probs = torch.nn.functional.softmax(result[0].data, dim=1).cpu().numpy()[0]
        classes = result[1].data.cpu().numpy()[0]
    else:       
        probs = torch.nn.functional.softmax(result[0].data, dim=1).numpy()[0]
        classes = result[1].data.numpy()[0]


    if labels:
        with open(labels, 'r') as f:
            cat_to_name = json.load(f)

        labels = list(cat_to_name.values())
        classes = [labels[x] for x in classes]
        
    model.train(mode=was_training)

    # Print only when invoked by command line 
    if args.inputs:
        print('The probabilities is:', list(zip(classes, probs)))
    
    return probs, classes

# Perform predictions from command line
if args.image and args.checkpoint:
    predict(args.image, args.checkpoint)