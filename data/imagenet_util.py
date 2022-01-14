import yaml
import os
import requests
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations
from pathlib import Path

class KeyNotFoundError(Exception):
    def __init__(self, cause, keys=None, visited=None):
        self.cause = cause
        self.keys = keys
        self.visited = visited
        messages = list()
        if keys is not None:
            messages.append("Key not found: {}".format(keys))
        if visited is not None:
            messages.append("Visited: {}".format(visited))
        messages.append("Cause:\n{}".format(cause))
        message = "\n".join(messages)
        super().__init__(message)


def synset2idx(path_to_yaml="data/imagenet_idx_to_synset.yaml"):
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    return dict((v,k) for k,v in di2s.items())

def give_synsets_from_indices(indices, path_to_yaml="data/imagenet_idx_to_synset.yaml"):
    synsets = []
    with open(path_to_yaml) as f:
        di2s = yaml.load(f)
    for idx in indices:
        synsets.append(str(di2s[idx]))
    print("Using {} different synsets for construction of Restricted Imagenet.".format(len(synsets)))
    return synsets


def str_to_indices(string):
    """Expects a string in the format '32-123, 256, 280-321'"""
    assert not string.endswith(","), "provided string '{}' ends with a comma, pls remove it".format(string)
    subs = string.split(",")
    indices = []
    for sub in subs:
        subsubs = sub.split("-")
        assert len(subsubs) > 0
        if len(subsubs) == 1:
            indices.append(int(subsubs[0]))
        else:
            rang = [j for j in range(int(subsubs[0]), int(subsubs[1]))]
            indices.extend(rang)
    return sorted(indices)

def download(url, local_path, chunk_size=1024):
    os.makedirs(os.path.split(local_path)[0], exist_ok=True)
    with requests.get(url, stream=True) as r:
        total_size = int(r.headers.get("content-length", 0))
        with tqdm(total=total_size, unit="B", unit_scale=True) as pbar:
            with open(local_path, "wb") as f:
                for data in r.iter_content(chunk_size=chunk_size):
                    if data:
                        f.write(data)
                        pbar.update(chunk_size)

def retrieve(
    list_or_dict, key, splitval="/", default=None, expand=True, pass_success=False
):
    """Given a nested list or dict return the desired value at key expanding
    callable nodes if necessary and :attr:`expand` is ``True``. The expansion
    is done in-place.
    Parameters
    ----------
        list_or_dict : list or dict
            Possibly nested list or dictionary.
        key : str
            key/to/value, path like string describing all keys necessary to
            consider to get to the desired value. List indices can also be
            passed here.
        splitval : str
            String that defines the delimiter between keys of the
            different depth levels in `key`.
        default : obj
            Value returned if :attr:`key` is not found.
        expand : bool
            Whether to expand callable nodes on the path or not.
    Returns
    -------
        The desired value or if :attr:`default` is not ``None`` and the
        :attr:`key` is not found returns ``default``.
    Raises
    ------
        Exception if ``key`` not in ``list_or_dict`` and :attr:`default` is
        ``None``.
    """
    keys = key.split(splitval)

    success = True
    try:
        visited = []
        parent = None
        last_key = None
        for key in keys:
            if callable(list_or_dict):
                if not expand:
                    raise KeyNotFoundError(
                        ValueError(
                            "Trying to get past callable node with expand=False."
                        ),
                        keys=keys,
                        visited=visited,
                    )
                list_or_dict = list_or_dict()
                parent[last_key] = list_or_dict

            last_key = key
            parent = list_or_dict

            try:
                if isinstance(list_or_dict, dict):
                    list_or_dict = list_or_dict[key]
                else:
                    list_or_dict = list_or_dict[int(key)]
            except (KeyError, IndexError, ValueError) as e:
                raise KeyNotFoundError(e, keys=keys, visited=visited)

            visited += [key]
        # final expansion of retrieved value
        if expand and callable(list_or_dict):
            list_or_dict = list_or_dict()
            parent[last_key] = list_or_dict
    except KeyNotFoundError as e:
        if default is None:
            raise e
        else:
            list_or_dict = default
            success = False
    if not pass_success:
        return list_or_dict
    else:
        return list_or_dict, success

class ImagePaths(Dataset):
    def __init__(self, paths, size=None, random_crop=False, labels=None,
                 uniform_dequantization=False, random_flip=False):
        self.size = size
        self.random_crop = random_crop
        self.uniform_dequantization = uniform_dequantization
        self.random_flip = random_flip

        self.labels = dict() if labels is None else labels
        self.labels["file_path_"] = paths
        self._length = len(paths)

        # if self.size is not None and self.size > 0:
        #     pipeline = list()
        #     pipeline.append(albumentations.SmallestMaxSize(max_size = self.size))
        #     if not self.random_crop:
        #         pipeline.append(albumentations.CenterCrop(height=self.size,width=self.size))
        #     else:
        #         pipeline.append(albumentations.RandomCrop(height=self.size,width=self.size))
        #     if self.random_flip:
        #         pipeline.append(albumentations.HorizontalFlip())
        #     self.preprocessor = albumentations.Compose(pipeline)
        # else:
        #     self.preprocessor = lambda **kwargs: kwargs

        # Perform preprocessing and transforms using torchvision -> only return clean image
        self.preprocessor = lambda **kwargs: kwargs

    def __len__(self):
        return self._length

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        if not image.mode == "RGB":
            image = image.convert("RGB")
        image = np.array(image).astype(np.uint8)
        image = self.preprocessor(image=image)["image"]
        if self.uniform_dequantization:
            image = image.astype(np.float32)
            image = image + np.random.uniform()/256.
        # image = (image/127.5 - 1.0).astype(np.float32)   # in range -1 ... 1 # Use torchvision normalization later
        return image

    def __getitem__(self, i):
        example = dict()
        example["image"] = self.preprocess_image(self.labels["file_path_"][i])
        for k in self.labels:
            example[k] = self.labels[k][i]
        return example


def is_prepared(root):
    return Path(root).joinpath(".ready").exists()


def mark_prepared(root):
    Path(root).joinpath(".ready").touch()