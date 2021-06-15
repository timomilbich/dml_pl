from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from utils.auxiliaries import instantiate_from_config
import batchminer as bmine


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size//2
        self.wrap = wrap

        ## Gather dataset configs
        if train is not None:
            self.dataset_configs["train"] = train
        if validation is not None:
            self.dataset_configs["validation"] = validation
        if test is not None:
            self.dataset_configs["test"] = test

        ## Init datasets
        self.setup()

        ## Init dataloaders
        if train is not None:
            ## Add datasampler if required
            if "data_sampler" in self.dataset_configs["train"].keys():

                config_datasampler = self.dataset_configs["train"]["data_sampler"]
                config_datasampler["params"]['image_dict'] = self.datasets["train"].dataset.image_dict
                config_datasampler["params"]['image_list'] = self.datasets["train"].dataset.image_list
                self.train_datasampler = instantiate_from_config(config_datasampler)

            self.train_dataloader = self._train_dataloader

        if validation is not None:
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.test_dataloader = self._test_dataloader

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          shuffle=True)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers)

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)