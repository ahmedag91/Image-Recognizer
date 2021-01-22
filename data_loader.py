
from torchvision import datasets, transforms
import torch 
import numpy as np
from PIL import Image as im
def load_image_dataset(dataset_dir: str):
    """
    This function takes a path to the root directory of a the dataset as a string variable. This directory should has three subfolders
    train, test, and valid folder. It returns a tuple of the train, valid, and test datasets.
    Arguments:
    ----------
    dataset_dir: an input string
    """
    dataset_dirs = [dataset_dir + '/train', dataset_dir + '/valid', dataset_dir + '/test']
    mean_per_channel, std_per_channel = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    crop_size = 224
    data_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(crop_size), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean_per_channel, std_per_channel)
                                    ])
    dataset = []
    image_data = []
    for dir in dataset_dirs:
        image_data.append(datasets.ImageFolder(dir, transform=data_transforms))
        dataset.append(torch.utils.data.DataLoader(datasets.ImageFolder(dir, transform=data_transforms), batch_size=64, shuffle=True))
        
    return ((images for images in image_data), (data for data in dataset))

def load_image(im_path:str):
    """
    This function returns a PIL.Image for a given path and file name as a string. 
    Arguments:
    ----------
    im_path: an input string
    """
    return im.open(im_path)

def process_image(image:im):
    """
        Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
        Arguments
        ---------
        image: a PIL.Image object
    """
   
    ### resize then crop
    max_size = 256
    image_resized = image.resize(size = tuple([max_size]*2))

    final_im_size = 224

    # Cropping criteria define the left, upper, right, lower corners as a list and convert them into tuples
    cropping = tuple([
                    (max_size - final_im_size)/2,
                    (max_size - final_im_size)/2,
                    (max_size + final_im_size)/2,
                    (max_size + final_im_size)/2
                    ])

    image_cropped = image_resized.crop(cropping)
    
    # Convert to np
    np_image = np.array(image_cropped, dtype = np.double)

    # then crop the image
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    
    for i in range(len(means)):
        np_image[:, :, i] = (np_image[:, :, i] - means[i])/stds[i]
        
    return np_image.reshape(-1, final_im_size, final_im_size)
