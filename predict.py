import argparse
import Model
import data_loader
import torch
import json
import numpy as np
parser = argparse.ArgumentParser(
    description='This for loading and testing a pretrained and saved network.'
)

parser.add_argument('--load_checkpoint', metavar = '', type = str, default = './my_model.pth', help = "Specify the dicrectory and the file name with extension '.pth' in which you wish to load your pretrained model parameters")

parser.add_argument('--top_k', metavar = '', type = int, default = 5, help = 'Specify the top k possible classes to which an image can be classified.')

parser.add_argument('--gpu', action = 'store_true', help = 'Specifiy whether you want to train your model on a CPU or a GPU by just writing --gpu. It chooses the GPU by default if it is available.')

parser.add_argument('--category_names', metavar = '', type = str, default = 'cat_to_name.json', help = 'Specify the directory and the name of json file that maps category number to the class name.')

parser.add_argument('--input', metavar = '', type = str, default = './flowers/test/5/image_05159.jpg', help = "Specify the path to an image with its name the '.jpg' extension.")

args = parser.parse_args()

# load the category names
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

# Load the saved model
model = Model.load_checkpoint(args.load_checkpoint)

# Load the Image
image = data_loader.load_image(args.input)

# Convert is to tensor
tensor_image = torch.from_numpy(data_loader.process_image(image)).float()


# Check whether to use GPU or CPU
if torch.cuda.is_available() and args.gpu:
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
#model = model.float()
#tensor_image = tensor_image.float()
print('{} is in use.'.format(device))

top_props, top_classes = Model.predict(tensor_image, model, topk=args.top_k, device = device)
print('Here are the top propabilities:\n{}'.format(top_props))
print('Here are the top classes:\n{}'.format(top_classes))
print('The corresponding class names:\n{}'.format(np.array([cat_to_name[top_class] for top_class in top_classes])))