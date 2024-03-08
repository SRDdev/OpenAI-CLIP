import os
import cv2
import torch
import albumentations as A
import config as config

#--------------------Dataset-----------------#
class CLIPDataset(torch.utils.data.Dataset):
    """
    define : CLIPDataset
    args : image_filenames, captions, tokenizer, transforms
    outputs : dataset instance for image-caption pairs
    note : prepares image filenames, captions, encoded captions, and transforms for dataset construction
    """

    def __init__(self, image_filenames, captions, tokenizer, transforms):
        """
        define : __init__ of CLIPDataset
        args : image_filenames, captions, tokenizer, transforms
        note : initializes image filenames, captions, encoded captions, and transforms
        """
        self.image_filenames = image_filenames
        self.captions = list(captions)
        self.encoded_captions = tokenizer(
            list(captions), padding=True, truncation=True, max_length=config.max_length
        )
        self.transforms = transforms

    def __getitem__(self, idx):
        """
        define : __getitem__ of CLIPDataset
        args : index idx
        outputs : item containing image and caption data
        note : retrieves image and caption data for a given index
        """
        item = {
            key: torch.tensor(values[idx])
            for key, values in self.encoded_captions.items()
        }

        image = cv2.imread(f"{config.image_path}/{self.image_filenames[idx]}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)['image']
        item['image'] = torch.tensor(image).permute(2, 0, 1).float()
        item['caption'] = self.captions[idx]

        return item

    def __len__(self):
        """
        define : __len__ of CLIPDataset
        args : None
        outputs : length of the dataset
        note : returns the length of the dataset
        """
        return len(self.captions)


def get_transforms(mode="train"):
    """
    define : get_transforms
    args : mode (default: "train")
    outputs : transforms instance based on mode
    note : returns a transforms instance based on the given mode
    """
    if mode == "train":
        return A.Compose(
            [
                A.Resize(config.size, config.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(config.size, config.size, always_apply=True),
                A.Normalize(max_pixel_value=255.0, always_apply=True),
            ]
        )