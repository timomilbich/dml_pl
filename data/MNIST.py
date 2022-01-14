import os
import os.path
import torch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

class DATA(Dataset):
    """`MNIST <http://yann.lecun.com/exdb/mnist/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``MNIST/training.pt``
            and  ``MNIST/processed/test.pt`` exist.
        train (bool, optional): If True, creates dataset from ``training.pt``,
            otherwise from ``test.pt``.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    training_file = 'training.pt'
    test_file = 'test.pt'

    def __init__(
            self,
            root,
            train = True,
            img_dim = 28,
            ooDML_split_id=-1,
            arch='resnet50',
    ):

        super(DATA, self).__init__()

        self.train = train  # training set or test set
        self.root = "/export/home/tmilbich/Datasets/MNIST/" if root is None else root
        self.n_classes = 10

        if self.train:
            self.dataset = torchvision.datasets.MNIST(self.root, train=True, download=True,
                                                 transform=torchvision.transforms.Compose(
                                                     [torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))]))
            self.image_dict = {
                'imgs': self.dataset.train_data,
                'labels': self.dataset.train_labels
            }
            self.transform = self.dataset.transform

        else:
            self.dataset = torchvision.datasets.MNIST(self.root, train=False, download=True,
                                                 transform=torchvision.transforms.Compose(
                                                     [torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))]))

            self.image_dict = {
                'imgs': self.dataset.test_data,
                'labels': self.dataset.test_labels
            }
            self.transform = self.dataset.transform

    def __getitem__(self, index: int):

        img, target = self.image_dict['imgs'][index], int(self.image_dict['labels'][index])

        # if self.transform is not None:
        #     img = self.transform(img)
        #
        # img = img.unsqueeze(0)/255

        return img, target

    def __len__(self):
        return len(self.image_dict['imgs'])

