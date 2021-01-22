# Transfer Learning for image classification using [PyTorch](https://pytorch.org/get-started/locally/)
This is a project code for a transfer learning project. In this project, image classifier is built with [PyTorch](https://pytorch.org/get-started/locally/) by applying the concept of transfer learning. This library deals with an image dataset with 102 classes. The library assumes that the dataset main folder has three subfolder train, valid, and test folders. Each folder contains 102 subfolder where each one has some images in `.jpg` format.

## Used Libraries 
Please check which of these require installation to run the library. Check also the documentation in case that you need.

- [Python](https://www.python.org/downloads/)
- [PyTorch](https://pytorch.org/get-started/locally/)
- [Pillow](https://pillow.readthedocs.io/en/stable/installation.html)
- [NumPy](https://numpy.org/)
- [argparse](https://docs.python.org/3/library/argparse.html)
- [matplotlib](https://matplotlib.org/)
- [Jupyter](https://jupyter.org/install)
- [Anaconda](https://www.continuum.io/downloads)

### Code

The code provided in the [`Image Classifier Project.ipynb`](https://github.com/ahmedag91/Image-Recognizer/blob/main/Image%20Classifier%20Project.ipynb) notebook file describes how everything was done. Moreover, it shows how the data is loaded and preprocessed. The notebook employed transfer learning on a pretrained [`VGG16`](https://pytorch.org/docs/stable/torchvision/models.html) neural network architecture. The notebook demonstrates how some layers of the pretrained neural network are frozen, while the other layers are replaced with a fully connected network and trained. There exists a [`cat_to_name.json`](https://github.com/ahmedag91/Image-Recognizer/blob/main/cat_to_name.json) file, which maps the folder name of each class of the dataset to an actual flower name. 

The rest of the project is divided into the following Python files:

- [`Model.py`](https://github.com/ahmedag91/Image-Recognizer/blob/main/Model.py): This file contains a Network class that implements a full-connected network for a given input. It also applies the transfer learning for a given pretrained model and a fully-connected network. Finally, it implements training and for a given neural-network architecture. Further details about the functions and the classes implemented in this file can be found at the documentation of file.
- [`data_loader.py`](https://github.com/ahmedag91/Image-Recognizer/blob/main/data_loader.py): This file has some implemented functions, which responsible for loading and preprocesing either a whole dataset or a single image with ``.jpg`` format.
- [`train.py`](https://github.com/ahmedag91/Image-Recognizer/blob/main/train.py): This a module is runnable by a command line prompt. It takes arguments from the user. These arguments are required to apply transfer learning for a given neural network and saves the newly trained architecture. To check what these arguments are and what kind of inputs they take, run the following command:

```
python train.py --help
```

An example of running the file from the command line and give all the required arguments is as follows

```
python train.py --data_dir 'flowers' --arch 'vgg16' --hidden_layers 4096 1000 --epochs 10 --gpu --save_checkpoint ./my_model.pth
```

- [`predict.py`](https://github.com/ahmedag91/Image-Recognizer/blob/main/predict.py)
- [`workspace_utils.py`](https://github.com/ahmedag91/Image-Recognizer/blob/main/workspace_utils.py): This one prevent prevents the session from being shutdown. This file is recommended to use if a cloud platform with time out is employed. It is recommended to use it during a long training time. An example of using it can be found at [`Image Classifier Project.ipynb`](https://github.com/ahmedag91/Image-Recognizer/blob/main/Image%20Classifier%20Project.ipynb) notebook.




