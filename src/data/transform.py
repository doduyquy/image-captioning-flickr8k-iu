from torchvision import transforms

def build_transforms():
    """ 
    - Train: lật xoay, ảnh ... (Data Augmentation) để mô hình học tổng quát hơn.
    - Val/Test: chuan hoa size
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((224, 224)),
        transforms.RandomHorizontalFlip(), # Lật ngang ngẫu nhiên
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Trị số chuẩn của ResNet
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Ép chuẩn đúng kích thước thay vì Crop ngẫu nhiên
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_test_transform
