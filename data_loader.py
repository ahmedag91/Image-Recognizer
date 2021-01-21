
from torchvision import datasets, transforms
import torch 
def load_image_dataset(dataset_dir: str):
    
    dataset_dirs = [dataset_dir + '/train', dataset_dir + '/valid', dataset_dir + '/test']
    mean_per_channel, std_per_channel = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    crop_size = 224
    data_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(crop_size), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean_per_channel, std_per_channel)
                                    ])
    data_set = []
    for dir in dataset_dirs:
        data_set.append(torch.utils.data.DataLoader(datasets.ImageFolder(dir, transform=data_transforms), batch_size=64, shuffle=True))
        
    return (data for data in data_set) 
    """

    train_dir = dataset_dir + '/train'
    valid_dir = dataset_dir + '/valid'
    test_dir = dataset_dir + '/test'
    mean_per_channel, std_per_channel = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

    data_transforms = transforms.Compose([
                                        transforms.RandomResizedCrop(224), 
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean_per_channel, std_per_channel)
                                    ])
                                    
    image_datasets = dict()
    image_datasets['train'] = datasets.ImageFolder(train_dir, transform=data_transforms)
    image_datasets['valid'] = datasets.ImageFolder(valid_dir, transform=data_transforms)
    image_datasets['test'] = datasets.ImageFolder(test_dir, transform=data_transforms)


    dataloaders = dict()
    dataloaders['train'] = torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True)
    dataloaders['valid'] = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
    dataloaders['test'] = torch.utils.data.DataLoader(image_datasets['test'], batch_size=64, shuffle=True)
    return [dataloaders['train'], dataloaders['valid'], dataloaders['test']]
    """


#switcher = {'one': 1, 'two': 2}
#print(switcher.get('one', 'invalid number'))

