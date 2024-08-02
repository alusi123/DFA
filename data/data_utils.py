from torchvision import transforms
import random
import torch
import numpy as np
from math import sqrt

def dataset_info(filepath):
    with open(filepath, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.strip().split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


def get_img_transform(train=False, image_size=224, crop=False, jitter=0):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    if train:
        if crop:
            img_transform = [transforms.RandomResizedCrop(image_size, scale=[0.8, 1.0])]
        else:
            img_transform = [transforms.Resize((image_size, image_size))]
        if jitter > 0:
            img_transform.append(transforms.ColorJitter(brightness=jitter,
                                                        contrast=jitter,
                                                        saturation=jitter,
                                                        hue=min(0.5, jitter)))
        img_transform += [transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          transforms.Normalize(mean, std)]
        img_transform = transforms.Compose(img_transform)
    else:
        img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    return img_transform


def get_pre_transform(image_size=224, crop=False, jitter=0):
    if crop:
        img_transform = [transforms.RandomResizedCrop(image_size, scale=[0.8, 1.0])]
    else:
        img_transform = [transforms.Resize((image_size, image_size))]
    if jitter > 0:
        img_transform.append(transforms.ColorJitter(brightness=jitter,
                                                    contrast=jitter,
                                                    saturation=jitter,
                                                    hue=min(0.5, jitter)))
    img_transform += [transforms.RandomHorizontalFlip(), lambda x: np.asarray(x)]
    img_transform = transforms.Compose(img_transform)
    return img_transform


def get_post_transform(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return img_transform

def get_digit_transform(image_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    img_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return img_transform


def get_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha

def get_centralized_spectrum(img):
    img_fft = np.fft.fft2(img)
    img_fft = np.fft.fftshift(img_fft)
    img_abs = np.abs(img_fft)
    img_pha = np.angle(img_fft)
    return img_abs, img_pha
