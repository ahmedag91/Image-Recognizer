import argparse

import torchvision
from torchvision.models.alexnet import alexnet
import Model
import data_loader
import torch
import torchvision
parser = argparse.ArgumentParser(
    description='This for loading and testing a pretrained and saved network.'
)

parser.add_argument('--load_checkpoint', metavar = '', type = str, default = './my_model.pth', help = 'Specify the dicrectory in which you wish to load your pretrained model parameters')

parser.add_argument('--top_k', metavar = '', type = int, default = 5, help = 'Specify the top k possible classes to which an image can be classified to.')

parser.add_argument('--gpu', action = 'store_true', help = 'Specifiy whether you want to train your model on a CPU or a GPU by just writing --gpu. It chooses the GPU by default if it is available.')

parser.add_argument('--category_names', metavar = '', type = str, default = 'cat_to_name.json', help = 'Specify the json file that maps category number to the class name.')

parser.add_argument('--input', metavar = '', type = str, default = './flowers/test/5/image_05159.jpg', help = 'Specify the path to an image with its name and extension.')

args = parser.parse_args()
