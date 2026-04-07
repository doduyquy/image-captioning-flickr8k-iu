import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from .flickr8k import load_captions, filter_valid_images, flatten_data, Flickr8kDataset
from .vocab import Vocabulary
from .collate import collate_fn
from .transform import build_transforms

def load_split_set(file_path):
    """ Đọc tập ID ảnh (vd: testImages.txt) thành một tập hợp (set) """
    if not os.path.exists(file_path):
        return None
        
    image_ids = set()
    with open(file_path, 'r') as file:
        for line in file:
            name = line.strip()
            if not name: 
                continue
            image_ids.add(name)
    return image_ids


def get_loaders_flickr8k(
    data_dir,
    image_dir,
    captions_file,
    vocab=None,
    batch_size=32,
    num_workers=2,
    freq_threshold=5,
    ):
    """ 
    Return:  get_loaders: train_loader, val_loader, test_loader 
    """
    
    # 1. Load và lọc ảnh lỗi
    print("Loading captions...")
    captions_dict = load_captions(captions_file)
    print("Filtering valid images...")
    captions_dict = filter_valid_images(image_dir, captions_dict)

    # 2. Đọc file chia tập (Split files)
    # Tùy dữ liệu của bạn lưu tên file như thế nào, ở đây tôi gọi chung là trainImages.txt
    train_set = load_split_set(os.path.join(data_dir, "Flickr_8k.trainImages.txt"))
    val_set = load_split_set(os.path.join(data_dir, "Flickr_8k.devImages.txt"))
    test_set = load_split_set(os.path.join(data_dir, "Flickr_8k.testImages.txt"))

    
    # 3. Gom Nhóm Dict -> 3 Dict riêng cho Train/Val/Test
    train_dict, val_dict, test_dict = {}, {}, {}
    for img, caps in captions_dict.items():
        if img in train_set:
            train_dict[img] = caps
        elif img in val_set:
            val_dict[img] = caps
        elif img in test_set:
            test_dict[img] = caps

    # 4. Phẳng hóa (Flatten data)!
    train_paths, train_caps = flatten_data(image_dir, train_dict)
    val_paths, val_caps = flatten_data(image_dir, val_dict)
    test_paths, test_caps = flatten_data(image_dir, test_dict)

    # 5. Build Vocab 
    # VOCAB ONLY FIT ON TRAIN SET. 
    if vocab is None:
        print("Building vocabulary from Train set...")
        vocab = Vocabulary(freq_threshold=freq_threshold)
        vocab.build_vocab(train_caps)

    # 6. Transform
    train_transform, val_test_transform = build_transforms()

    # 7. Khởi tạo 3 Dataset độc lập
    print("Building datasets...")
    train_dataset = Flickr8kDataset(train_paths, train_caps, vocab, transform=train_transform)
    val_dataset = Flickr8kDataset(val_paths, val_caps, vocab, transform=val_test_transform)
    test_dataset = Flickr8kDataset(test_paths, test_caps, vocab, transform=val_test_transform)

    # 8. Khởi tạo DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, collate_fn=collate_fn)
    # Val/Test: (shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, 
                             num_workers=num_workers, collate_fn=collate_fn)

    print(f"Loader done: Train ({len(train_dataset)}), Val ({len(val_dataset)}), Test ({len(test_dataset)})")
    
    return (train_loader, val_loader, test_loader), vocab






if __name__ == "__main__":
    import sys
    # Đảm bảo có thể import được src.utils nếu chạy trực tiếp file này
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.utils.config import load_config

    print("=== Testing DataLoader với cấu hình chuẩn từ Config ===")
    
    # 1. Load bằng hàm config chuẩn của source
    config = load_config(model="lstm", env="local")
    
    data_dir = config["flickr8k"]["data_dir"]
    image_dir = config["flickr8k"]["image_dir"]
    captions_file = config["flickr8k"]["captions_file"]
    
    print(f"- Data Dir: {data_dir}")
    print(f"- Image Dir: {image_dir}")
    print(f"- Captions File: {captions_file}\n")

    try:
        # 2. Gọi hàm mồi dữ liệu thực tế
        loaders, vocab = get_loaders_flickr8k(
            data_dir=data_dir,
            image_dir=image_dir,
            captions_file=captions_file,
            vocab=None,
            batch_size=config["data"]["batch_size"],
            num_workers=config["data"]["num_workers"],
            freq_threshold=config["data"]["freq_threshold"],
        )
        
        train_loader, val_loader, test_loader = loaders
        print("\n=> Khởi tạo DataLoader thành công!")
        print(f"Kích thước từ điển (Vocab): {len(vocab)} từ")
        
        # 3. batch ...
        images, captions = next(iter(train_loader))
        print("\nThông tin 1 Batch từ Train Loader:")
        print(f" - Hình ảnh (Images)   : {images.shape}")
        print(f" - Văn bản (Captions)  : {captions.shape}")

    except FileNotFoundError as e:
        print("[Lỗi] Không tìm thấy dữ liệu vật lý trên máy tính hiện tại của bạn!")
        print(f"   Chi tiết: {e}")
