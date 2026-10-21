import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

def get_train_transforms(image_size=224):
    """Augmentations simulating social media compression and artifacts."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.ImageCompression(quality_lower=50, quality_upper=95, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_val_transforms(image_size=224):
    """Standard evaluation transformation pipeline."""
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
