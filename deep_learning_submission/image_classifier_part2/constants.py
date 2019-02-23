from torchvision import models

# dataset names
DATASET_NAMES = ['train', 'test', 'validation']

# Normalize to this mean values
MEAN_NORMALIZE = [0.485, 0.456, 0.406]

# Normalize to these std values
STD_NORMALIZE = [0.229, 0.224, 0.225]

# values in degrees to apply for random rotation for image transform 
TRANSFORM_ROTATION_DEGREE = 50

# Crop size in pixels for image transform
TRANSFORM_CROP_SIZE = 224

# Horizontal flip probabilty for image transformation
TRANSFORM_HFLIP_PROB = 0.5

# Resize image to these pixels
TRANSFORM_RESIZE = 255

# # Image dirs:
# DATA_DIR = 'flowers'

IMAGE_DIRS = {
    'train': '/train',
    'test': '/test',
    'validation': '/valid'
}
           
    
# Available VGG model names to build on
VGG_MODEL_NAMES = ['vgg11', 'vgg13', 'vgg16', 'vgg19']

VGG_MODELS = {
    'vgg11' : models.vgg11(pretrained=True),
    'vgg13' : models.vgg13(pretrained=True),
    'vgg16' : models.vgg16(pretrained=True),
    'vgg19' : models.vgg19(pretrained=True),
}

# Print the train loss, validation loss and validation accuracy in each how many batches:
PRINT_EVERY = 10