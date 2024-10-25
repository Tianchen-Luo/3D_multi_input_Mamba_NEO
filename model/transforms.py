import torch
from torchvision import transforms

basic_augmentation_pipeline = transforms.Compose([
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15)
])

def add_gaussian_noise(img, mean=0, std=20):
 
    noise = torch.randn(img.size()) * std + mean
    noisy_img = img + noise
    return noisy_img.clamp(0, 255)   

 
moderate_noise_and_large_translation_augmentation = transforms.Compose([
    transforms.Lambda(lambda img: add_gaussian_noise(img, std=20)),
    transforms.RandomAffine(degrees=(-6, 6), translate=(0.25, 0.75))
])

light_noise_and_small_translation_augmentation = transforms.Compose([
    transforms.Lambda(lambda img: add_gaussian_noise(img, std=20)),
    transforms.RandomAffine(degrees=(0, 0.5), translate=(0, 0.1))
])

slight_noise_and_moderate_translation_augmentation = transforms.Compose([
    transforms.Lambda(lambda img: add_gaussian_noise(img, std=20)),
    transforms.RandomAffine(degrees=(0.5, 0.5), translate=(0.1, 0.1))
])

horizontal_flip_and_rotation_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=(-60, 60)),
    transforms.RandomAffine(degrees=0, scale=(0.8, 1.2))
])
