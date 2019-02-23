import os
import json
import argparse
import torch 
import numpy as np
from torchvision import datasets, transforms
from PIL import Image
import constants


def predict_argparser():
    """
        Argparser for predict.py
    """
    
    parser = argparse.ArgumentParser(description='Predict Argparser')
    
    parser.add_argument('input', action='store', help='path to image to predict')
    parser.add_argument('checkpoint', action='store', help='checkpoint name, search path is set to ./saved_models ')
    parser.add_argument('--top_k', action='store', type=int, default=1, help='Top K mostly likely classes')
    parser.add_argument('--category_names', action='store', default=None, help='Mapping of categories to real names')
    parser.add_argument('--gpu', action='store_true', default=False, help='Enable GPU (cuda) for prediction')
    
    return parser.parse_args()

def train_argparser():
    """
        Argparser for train.py
    """
    parser = argparse.ArgumentParser(description='Train Argparser')
    
    parser.add_argument('data_dir', action='store', help='path to directory where train validation and test images are stored')
    parser.add_argument('--learning_rate', action='store', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--hidden_units', action='store', type=int, default=2048, help='Number of perceptrons in the hidden layer')
    parser.add_argument('--epochs', action='store', type=int, default=3, help='How many times we need to iterate over the entire training set')
    parser.add_argument('--gpu', action='store_true', default=False, help='Enable GPU (cuda) for training')
    parser.add_argument('--arch', action='store', default='vgg11', help='Pre-trained model architecture')
    parser.add_argument('--save_dir', action='store', default='saved_models', help='path to directory where we save our trained model')
    
    return parser.parse_args()
    
def save_model(model, epoch, optimizer, train_data, save_dir, name='trained_model.pth'):
    """
        Save trained model to disk
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'class_to_idx': train_data.class_to_idx,
        'classifier': model.classifier,
    }
    
    # create dir if doesn't exist 
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    torch.save(checkpoint, save_dir + '/' + name)

def load_model(checkpoint_name, save_dir='saved_models', train_components=False):
    """
        Load the model trained for the task
        It uses features from pretrained vgg model given in checkpoint arg 
        and was trained with a custom fully connected layer (classifier)
        it returns `epoch`, `optimizer` and `criterion` for further training 
        if train_components=True, otherwise it only returns model 
        
        checkpoint: VGGmodel name, valid names are: vgg11, vgg13, vgg15, vgg19
        save_dir: it assumes checkpoints are saved under save_dir
        train_components: return only model if set to False otherwise return training params
    """
    
    if checkpoint_name not in constants.VGG_MODEL_NAMES:
        raise ValueError("Invalid checkpoint name, valid pretrained models are: {}".format(constants.VGG_MODEL_NAMES))
    
    checkpoint_path = '{}/trained_model_{}.pth'.format(save_dir, checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise IOError('Cannot find checkpoint at: {}. Make sure you have a trained model here'.format(checkpoint_path))
    
    # load the saved model dict 
    checkpoint = torch.load(checkpoint_path)
    
    # Load VGG model from torchvision
    model = vgg_model_select(checkpoint_name)
    for param in model.parameters():
            param.requires_grad = False
    
    # replace fully connected layer with our loaded model
    model.classifier = checkpoint['classifier']
    
    # Class mapping
    model.class_to_idx =  checkpoint['class_to_idx']
    
    # Load weights for the model
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # If the user only asks for model, not for other params like optimizer, epoch etc..
    # return model, else return a tuple of model, epoch, optimizer, criterion
    if not train_components:
        return model
        
    # create optimizer and load state dict
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # criterion used for training
    criterion = nn.NLLLoss()
    
    epoch = checkpoint['epoch']
    
    return model, epoch, optimizer, criterion 

def process_image(image):
    """ 
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns a numpy array
    """
    
    # Assuming image is a PIL.Image object, let's not modify the source image,
    # but create a copy of it, so function has no side effects for the image
    img = image.copy()
    
    # we can make use of torchvision.transform
    # Define transfromations
    
    transform = transforms.Compose([
        transforms.Resize(size=constants.TRANSFORM_RESIZE),
        transforms.CenterCrop(size=constants.TRANSFORM_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=constants.MEAN_NORMALIZE, 
            std=constants.STD_NORMALIZE)
    ])
    
    # Note that transform variable defined above is now actually a callable, since 
    # transforms.Compose implements `def __call__(self, img):` 
    # Therefore we can use PIL Image as an input to transform callable.
    # Apply transform to PIL image, it returns a tensor
    img_tensor = transform(img)
    
    # convert tensor to numpy array and return
    return img_tensor.numpy()


def label_maps(file_path='cat_to_name.json'):
    """
        Human friendly flower label mappings
    """
    with open(file_path, 'r') as f:
        return json.load(f)

def vgg_model_select(name):
    """
        Select a pretrained VGG model based on a given name
    """
    
    if name not in constants.VGG_MODEL_NAMES:
        raise ValueError("Invalid VGG model '{0}', valid pretrained models are: {1}".format(name, constants.VGG_MODEL_NAMES))

    return constants.VGG_MODELS[name]

def data_loader(data_dir, dataset='train', batch_size=64):
    """
        Helper function to get data loader and data for different datasets
        
        data_dir: main directory where we keep oour dataset, in our case this is 'flowers'
        dataset: choose one of 'train', 'test', 'validation'
        batch_size = how many images data loader should return in each iteration , default=64
    """
    
    # make it case insensitive 
    dataset = dataset.lower()
    
    if dataset not in constants.DATASET_NAMES:
        raise ValueError('Invalid dataset parameter, valid dataset options are: {}'.format(constants.DATASET_NAMES)) 
    
    # Get transform to apply to dataset 
    transform = data_transforms(dataset)
    
    # Load the datasets with ImageFolder
    directory = data_dir + constants.IMAGE_DIRS[dataset]
    data = datasets.ImageFolder(directory, transform=transform)
    # Using the image dataset and the trainform, return the dataloader
    # just for train dataset we need to shuffle the images, keep false for
    # train and validation
    shuffle = True if dataset == 'train' else False
    
    data_loader = torch.utils.data.DataLoader(data, batch_size, shuffle=shuffle)
    
    return data, data_loader
    
def data_transforms(dataset):
    """
    choose the data transformations we will apply to a given dataset 
    
    dataset: could be 'train' 'test' or 'validation'
    return: torchvision transforms to apply to a given 'dataset' name 
    """
   
    if dataset == 'train':
        return transforms.Compose([
            transforms.RandomRotation(degrees=constants.TRANSFORM_ROTATION_DEGREE),
            transforms.RandomResizedCrop(size=constants.TRANSFORM_CROP_SIZE),
            transforms.RandomHorizontalFlip(p=constants.TRANSFORM_HFLIP_PROB),
            transforms.ToTensor(),
            transforms.Normalize(mean=constants.MEAN_NORMALIZE, 
                                 std=constants.STD_NORMALIZE)
        ])
    
    # If dataset is test or validation apply this transformation
    return transforms.Compose([
        transforms.Resize(size=constants.TRANSFORM_RESIZE),
        transforms.CenterCrop(size=constants.TRANSFORM_CROP_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=constants.MEAN_NORMALIZE, 
                             std=constants.STD_NORMALIZE)
    ])

if __name__ == '__main__':
#     print(data_loader(data_dir='flowers', dataset='validation', batch_size=64))
    print(label_map())