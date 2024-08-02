from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFilter
from data.data_utils import *
import random
import cv2
import os
import torch
import torch.nn.functional as F

# image_dir = "/home/q23301273"

class DGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None):
        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        img_name = self.names[index]
        img_name = self.args.image_dir + img_name
        img = Image.open(img_name).convert('RGB')
        if self.transformer is not None:
            img = self.transformer(img)
        label = self.labels[index]
        return img, label



class FourierDGDataset(Dataset):
    def __init__(self, args, names, labels, transformer=None, from_domain=None, alpha=1.0, random=True):

        self.args = args
        self.names = names
        self.labels = labels
        self.transformer = transformer
        self.post_transform = get_post_transform()
        self.from_domain = from_domain
        self.alpha = alpha
        self.random = random
        
        self.flat_names = []
        self.flat_labels = []
        self.flat_domains = []
        for i in range(len(names)):
            self.flat_names += names[i]
            self.flat_labels += labels[i]
            self.flat_domains += [i] * len(names[i])
        assert len(self.flat_names) == len(self.flat_labels)
        assert len(self.flat_names) == len(self.flat_domains)

    def __len__(self):
        return len(self.flat_names)

    def __getitem__(self, index):
        img_name = self.flat_names[index]
        label = self.flat_labels[index]
        domain = self.flat_domains[index]

        img_name = self.args.image_dir + img_name
        img = Image.open(img_name).convert('RGB')
        img = self.transformer(img) 
        img_list = [img]
        label_list = [label]
        domain_list = [domain]

        domains = list(range(len(self.names)))
        domains.remove(domain)
        for domain_idx in domains:
            img_idx = random.randint(0, len(self.names[domain_idx])-1)
            img_name_sampled = self.names[domain_idx][img_idx]
            img_name_sampled = self.args.image_dir + img_name_sampled
            img_sampled = Image.open(img_name_sampled).convert('RGB')
            img_sampled = self.transformer(img_sampled)
            label_sampled = self.labels[domain_idx][img_idx]
            img_list.append(img_sampled)
            label_list.append(label_sampled)
            domain_list.append(domain_idx)

        img_list = colorful_spectrum_mix(img_list, alpha=self.alpha, random=self.random)
        image_list = [self.post_transform(image) for image in img_list]
        label_list = label_list * 2
        domain_list = domain_list * 2
        
        return (image_list, label_list), domain_list
        
    
def colorful_spectrum_mix(img_list, alpha, ratio=1.0, random=True):
    """Input image size: ndarray of [H, W, C]"""
    lam = alpha
    if random is True:
        lam = np.random.uniform(0, alpha)

    img1 = img_list[0]
    img2 = img_list[1]
    img3 = img_list[2]
    assert img1.shape == img2.shape
    assert img1.shape == img3.shape

    h, w, c = img1.shape
    h_crop = int(h * sqrt(ratio)) 
    w_crop = int(w * sqrt(ratio)) 
    h_start = h // 2 - h_crop // 2 
    w_start = w // 2 - w_crop // 2 

    img1_fft = np.fft.fft2(img1, axes=(0, 1))
    img2_fft = np.fft.fft2(img2, axes=(0, 1))
    img3_fft = np.fft.fft2(img3, axes=(0, 1))
    img1_abs, img1_pha = np.abs(img1_fft), np.angle(img1_fft)
    img2_abs, img2_pha = np.abs(img2_fft), np.angle(img2_fft)
    img3_abs, img3_pha = np.abs(img3_fft), np.angle(img3_fft)

    img1_abs = np.fft.fftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.fftshift(img2_abs, axes=(0, 1))
    img3_abs = np.fft.fftshift(img3_abs, axes=(0, 1))

    img1_abs_ = np.copy(img1_abs)
    img2_abs_ = np.copy(img2_abs)
    img3_abs_ = np.copy(img3_abs)
    img1_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img2_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img1_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]
    img2_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img3_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img2_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img3_abs[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        lam * img1_abs_[h_start:h_start + h_crop, w_start:w_start + w_crop] + (1 - lam) * img3_abs_[
                                                                                          h_start:h_start + h_crop,
                                                                                          w_start:w_start + w_crop]

    img1_abs = np.fft.ifftshift(img1_abs, axes=(0, 1))
    img2_abs = np.fft.ifftshift(img2_abs, axes=(0, 1))
    img3_abs = np.fft.ifftshift(img3_abs, axes=(0, 1))

    img21 = img1_abs * (np.e ** (1j * img1_pha))
    img32 = img2_abs * (np.e ** (1j * img2_pha))
    img13 = img3_abs * (np.e ** (1j * img3_pha))
    img21 = np.real(np.fft.ifft2(img21, axes=(0, 1)))
    img32 = np.real(np.fft.ifft2(img32, axes=(0, 1)))
    img13 = np.real(np.fft.ifft2(img13, axes=(0, 1)))
    img21 = np.uint8(np.clip(img21, 0, 255))
    img32 = np.uint8(np.clip(img32, 0, 255))
    img13 = np.uint8(np.clip(img13, 0, 255))

    img_list += [img21, img32, img13]
    return img_list

def get_dataset(args, path, train=False, image_size=224, crop=False, jitter=0, config=None):
    names, labels = dataset_info(path) 
    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
    img_transform = get_img_transform(train, image_size, crop, jitter)
    return DGDataset(args, names, labels, img_transform)


def get_fourier_dataset(args, path, image_size=224, crop=False, jitter=0, from_domain='all', alpha=1.0, random=True, config=None):
    assert isinstance(path, list) 
    names = []
    labels = []
    for p in path:
        name, label = dataset_info(p)
        names.append(name)
        labels.append(label)

    if config:
        image_size = config["image_size"]
        crop = config["use_crop"]
        jitter = config["jitter"]
        from_domain = config["from_domain"]
        alpha = config["alpha"]
        # random=config["random"]

    img_transform = get_pre_transform(image_size, crop, jitter)
    return FourierDGDataset(args, names, labels, img_transform, from_domain, alpha, random)






