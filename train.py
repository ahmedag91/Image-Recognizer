import argparse

parser = argparse.ArgumentParser(
    description='This for training a network you choose.\n First, you need an existing available architecture to load such as VGG16 or VGG13, etc.\n Some layer will be freezed, whereas the last layers are dropped and replaced by hidden layers from your choice.'
)

#Taking the dataset directory as input from the user
parser.add_argument('--data_dir', metavar = '', type = str, default='flowers/', help = 'Specify the root directory of the dataset')

#Taking the dataset directory as input from the user
parser.add_argument('--save_dir', metavar = '', type = str, default='my_model.pth', help = 'Specify the root directory including the file name with .pth extension')

# Taking the learning rate from the user as float variable from the user
parser.add_argument('--learning_rate', metavar = '', type = float, default = 0.005, help = 'Specify the learning rate of the SGD optimizer that will be used')

# Take the model name architecture for more information about these architectures visit pytorch.org
parser.add_argument('--arch', metavar = '', type = str, default = 'vgg16', help = 'Specify the neural network architecture that will be used. Choose either vgg13, vgg16, or alexnet. The default is vgg16')

# Take the hidden layer dimensions as inputs
parser.add_argument('--hidden_layers', metavar = '', type = int, nargs = '+', default = [4096], help = 'Specify the dimensions of the hidden layers as list. The default has only a single element with 4096 neurons')

# Take the number of epochs as inputs
parser.add_argument('--epochs',  metavar= '', type = int, defauls = 10, help ='The number of training epochs the default value is 10')

# Take the device type you wish to train your model on
parser.add_argument('--gpu', action = 'store_true', help = 'Specifiy whether you want to train your model on a CPU or a GPU by just writing --gpu. It chooses the GPU by default if it is available')

parser.add_argument('--dropout', metavar = '', type = float, default = 0.2, help = 'Specify the dropout rate that will be used for the fully connected layers, which you will specify their dimensions. The default value is 0.2')

args = parser.parse_args()