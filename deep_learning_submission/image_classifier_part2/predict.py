import argparse
import torch
import utils
import numpy as np
from PIL import Image

def predict(args):
    """
        Predict the class (or classes) of an image using a trained deep learning model.
    """
    
     # Get args
    image_path = args.input                 # Path to prediction image
    checkpoint = args.checkpoint            # vgg model to load
    topk = args.top_k                       # return top K prediction classes
    category_names = args.category_names    # Return actual class names
    gpu = args.gpu                          # True If GPU is used for prediction
    
    # load model from checkpoint
    model = utils.load_model(checkpoint)
    
    # Process image (do transformations) for prediction
    image = Image.open(image_path)
    processed = utils.process_image(image)
    
    # Expand numpy array dimention and convert to tensor
    processed  = np.expand_dims(processed,axis=0)
    X = torch.from_numpy(processed)
    
    # Use GPU if requested
    device = torch.device("cuda" if gpu else "cpu")
    
    # move model to device and set to eval mode
    model.to(device)
    model.eval()
    
    # Run a forward pass and predict classes  
    with torch.no_grad():
        
        # Move inputs to device(GPU)
        X = X.to(device)

        logps = model.forward(X)
        ps = torch.exp(logps)
        probs, classes = ps.topk(topk, dim=1)
    
    # Convert Tensor to numpy array 
    if probs.device.type == 'cuda':
        probs = probs.data.cpu().numpy().squeeze()
        classes = classes.data.cpu().numpy().squeeze()
    else:
        probs = probs.data.numpy().squeeze()
        classes = classes.data.numpy().squeeze()
    
    # Convert labels to actual flower names if category_names provided
    # Otherwise return classes
    if not category_names:
        
        reverse_dict = dict()
        for k, v in model.class_to_idx.items():
            reverse_dict[v] = k
            
        if topk > 1:
            classes = [reverse_dict[elem] for elem in classes]
        else:
            classes = reverse_dict[int(classes)]
        return probs.tolist(), classes
    
    # Category names provided do translation
    cat_to_name = utils.label_maps(category_names)

    reverse_dict = dict()
    for k, v in model.class_to_idx.items():
        reverse_dict[v] = cat_to_name[k]
        
    if topk > 1:
        classes = [reverse_dict[elem] for elem in classes]
    else:
        classes = reverse_dict[int(classes)]
    return probs.tolist(), classes
    
if __name__ == "__main__":
    args = utils.predict_argparser()
    print(predict(args))