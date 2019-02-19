import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data as data


import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse

# Define command line arguments
arguments_parser = argparse.ArgumentParser(description='Training a network')
arguments_parser.add_argument('--data_dir', action="store",type=str, help='Set a Path to database ')
arguments_parser.add_argument('--gpu', action='store_true', help='Apply GPU')
arguments_parser.add_argument('--epochs', type=int, action="store",help='Number of epochs')
arguments_parser.add_argument('--arch', type=str, action="store",help='Network architecture')
arguments_parser.add_argument('--learning_rate', type=float, action="store",help='A learning rate')
arguments_parser.add_argument('--hidden_units', type=int, action="store",help='Number of hidden layers')
arguments_parser.add_argument('--checkpoint', type=str, action="store",help='Create a checkpoint of trained model to saved in a file')



args, _ = arguments_parser.parse_known_args()

# loads and prepared a data in a pre-trained model
def load_arch(arch='vgg19', labels=102, hidden_layers=6000):
    # Load a pre-trained model
    if arch=='vgg19':
        # Load a pre-trained model
        model = models.vgg19(pretrained=True)
    elif arch=='alexnet':
        model = models.alexnet(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)               
    else:
        raise ValueError('Unknown pretrained model network', arch)
        
    # Freeze its parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Remove the last layer
    remove = list(model.classifier.children())[:-1]
  
    # Number of filters in the bottleneck layer
    bottleneck_layer = model.classifier[len(remove)].in_features

    # Extend the existing architecture with new layers
    remove.extend([
        nn.Dropout(),
        nn.Linear(bottleneck_layer, hidden_layers),
        nn.ReLU(True),
        nn.Dropout(),
        nn.Linear(hidden_layers, hidden_layers),
        nn.ReLU(True),
        nn.Linear(hidden_layers, labels),
        
    ])
    
    model.classifier = nn.Sequential(*remove)

    return model



# Trains a model network
def train_model(image_datasets, arch='vgg19', hidden_layers=4096, epochs=25, learning_rate=0.001, gpu=False, checkpoint=''):
    # Use command line values when specified
    if args.arch:
        arch = args.arch     
        
    if args.hidden_layers:
        hidden_layers = args.hidden_layers

    if args.epochs:
        epochs = args.epochs
            
    if args.learning_rate:
        learning_rate = args.learning_rate

    if args.gpu:
        gpu = args.gpu

    if args.checkpoint:
        checkpoint = args.checkpoint        
        
    # Using the image datasets, define the dataloaders
    dataloaders = {
        x: data.DataLoader(image_datasets[x], batch_size=4, shuffle=True, num_workers=2)
        for x in list(image_datasets.keys())
    }
 
    # Est. dataset sizes.
    dataset_sizes = {
        x: len(dataloaders[x].dataset) 
        for x in list(image_datasets.keys())
    }    

        
    print('Network architecture:', arch)
    print('Number of hidden layers:', hidden_layers)
    print('Number of epochs:', epochs)
    print('Learning rate:', learning_rate)

    # Load the model     
    num_labels = len(image_datasets['train'].classes)
    model = load_arch(arch=arch, num_labels=num_labels, hidden_layers=hidden_layers)

    # Use gpu if selected and available
    if gpu and torch.cuda.is_available():
        print('Using GPU for training')
        device = torch.device("cuda:0")
        model.cuda()
    else:
        print('Using CPU for training')
        device = torch.device("cpu")     

                
    # Defining criterion, optimizer and scheduler
    # Observe that only parameters that require gradients are optimized
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(list(filter(lambda p: p.requires_grad, model.parameters())), lr=learning_rate, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)    
        


    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:                
                inputs = inputs.to(device)
                labels = labels.to(device)
                

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()


    print('The validation training: {:4f}'.format(best_acc))

    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Store class_to_idx into a model property
    model.class_to_idx = image_datasets['train'].class_to_idx
    
    # Save checkpoint if requested
    if checkpoint:
        print ('Saving checkpoint to:', checkpoint) 
        checkpoint_dict = {
            'arch': arch,
            'class_to_idx': model.class_to_idx, 
            'state_dict': model.state_dict(),
            'hidden_layers': hidden_layers
        }
        
        torch.save(checkpoint_dict, checkpoint)
    
    # Return the model
    return model


# Train model if invoked from command line
if args.data_dir:    
    # Default transforms for the training, validation, and testing sets
    data_transforms = {
        'training': transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'validing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ]),
        'testing': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
         ])
    }
    
    # Load the datasets with ImageFolder
    image_datasets = {
        x: datasets.ImageFolder(root=args.data_dir + '/' + x, transform=data_transforms[x])
        for x in list(data_transforms.keys())
    }
        
    train_model(image_datasets) 