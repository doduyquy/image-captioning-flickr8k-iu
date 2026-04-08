from torchvision import transforms

def build_transforms():
    """ 
    - Train: lật xoay, ảnh ... (Data Augmentation) để mô hình học tổng quát hơn.
    - Val/Test: chuan hoa size
    """
   # Tăng cường dữ liệu trong dataloader.py
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # Cắt ngẫu nhiên
        transforms.RandomHorizontalFlip(),                   # Lật ảnh ngang
        transforms.ColorJitter(brightness=0.2, contrast=0.2), # Đổi màu nhẹ
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Ép chuẩn đúng kích thước thay vì Crop ngẫu nhiên
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_test_transform
