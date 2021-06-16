import os
import os.path
import torch
from torch.utils.data import Dataset
from .basic_dml_dataset import BaseDataset


class CUB200DATA(Dataset):
    """`CUB200 dataset.

    Args:
        root (string): Root directory of data.
        train (bool, optional): Specifies type of data split (train or validation).
        arch (string, optional): Type of network architecture used for training, influences choice of transformations
            applied when sampling batches.

    """

    def __init__(
            self,
            root,
            train = True,
            arch = 'resnet50',
            ):

        super(CUB200DATA, self).__init__()

        self.train = train  # training set or test set
        self.root = root

        image_sourcepath = root + '/images'
        image_classes = sorted([x for x in os.listdir(image_sourcepath) if '._' not in x],
                               key=lambda x: int(x.split('.')[0]))
        total_conversion = {int(x.split('.')[0]) - 1: x.split('.')[-1] for x in image_classes}
        image_list = {int(key.split('.')[0]) - 1: sorted(
            [image_sourcepath + '/' + key + '/' + x for x in os.listdir(image_sourcepath + '/' + key) if '._' not in x])
                      for key in image_classes}
        image_list = [[(key, img_path) for img_path in image_list[key]] for key in image_list.keys()]
        image_list = [x for y in image_list for x in y]

        ### Dictionary of structure class:list_of_samples_with_said_class
        image_dict = {}
        for key, img_path in image_list:
            if not key in image_dict.keys():
                image_dict[key] = []
            image_dict[key].append(img_path)

        ### Use the first half of the sorted data as training and the second half as test set
        keys = sorted(list(image_dict.keys()))
        train, test = keys[:len(keys) // 2], keys[len(keys) // 2:]

        ###
        train_conversion = {i: total_conversion[key] for i, key in enumerate(train)}
        test_conversion = {i: total_conversion[key] for i, key in enumerate(test)}

        ###
        train_image_dict = {key: image_dict[key] for key in train}
        test_image_dict = {key: image_dict[key] for key in test}

        ###
        if self.train:
            train_dataset = BaseDataset(train_image_dict, arch)
            train_dataset.conversion = train_conversion
            self.dataset = train_dataset
            print(f'Dataset Setup (Train)): #Classes: {len(train_image_dict)}')
        else:
            test_dataset = BaseDataset(test_image_dict, arch, is_validation=True)
            test_dataset.conversion = test_conversion
            self.dataset = test_dataset
            print(f'Dataset Setup (Val)): #Classes: {len(test_image_dict)}\n')

        # eval_dataset = BaseDataset(train_image_dict, opt, is_validation=True)
        # eval_train_dataset = BaseDataset(train_image_dict, opt, is_validation=False)
        # eval_dataset.conversion = test_conversion
        # eval_train_dataset.conversion = train_conversion

    def __getitem__(self, idx):
        return self.dataset.__getitem__(idx)


    def __len__(self):
        return len(self.dataset)

