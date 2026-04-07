import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image
from PIL.Image import fromarray

class Flickr8kDataset(Dataset):
    def __init__(self, image_paths, captions, vocab, transform=None):
        self.image_paths=image_paths
        self.captions=captions  
        self.vocab=vocab # đổi tượng lớp vocab đã tạo
        self.transform=transform#các phép biến đổi ảnh
        #-> nhiệm vụ là lưu trữ thông tin cần thiết

    def __len__(self):
        return len(self.image_paths)
    #-> cho pytỏch biết bộ dữ liệu này có tổng cộng bao nhiêu cặp (ảnh <-> caption). Khi train thfi pytỏch sẽ dựa vào con số này để biết khi nào thf hết 1 vòng

    def __getitem__(self, idx):
        """Return: (image_tensor, caption_tensor)"""
        image = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        caption = self.captions[idx]

        tokens = self.vocab.numericalize(caption)

        numericalized = [self.vocab.stoi["<start>"]]
        numericalized += tokens
        numericalized.append(self.vocab.stoi["<end>"])

        return image, torch.tensor(numericalized)


# Đọc file .txt hoặc .csv và gom tất cả các mô tả (captions) của cùng một ảnh vào một nhóm.
def load_captions(captions_file):
    captions_dict={} #key là ảnh, value là dsach các câu mô tả của từng ảnh
    with open(captions_file, "r",encoding="utf-8") as f: #mở file
        for line in f:    #duyệt từng file
            line=line.strip()
            if len(line)==0:#bỏ qua các dòng trống trong file txt
                continue

            # Skip common CSV header lines.
            lowered = line.lower()
            if lowered in {"image,caption", "image_name,comment_number,comment"}:
                continue

            # Support both formats:
            # 1) captions.txt: image.jpg,caption text
            # 2) Flickr8k.token.txt: image.jpg#0\tcaption text
            if "\t" in line:
                left, cap = line.split("\t", 1)
                img = left.split("#", 1)[0]
            elif "," in line:
                img, cap = line.split(",", 1)
            else:
                continue

            img = img.strip()
            cap = cap.lower().strip()
            if img not in captions_dict:
                captions_dict[img]=[] #nếu ảnh đã xuất hiện -> tạo 1 ds trống cho nó
            captions_dict[img].append(cap)  # thêm câu mô tả vào ds của ảnh tương ứng

    if not captions_dict:
        raise ValueError(f"Khong doc duoc caption nao tu file: {captions_file}")

    return captions_dict


### ---- Utils function for Flickr8k ---- ###
#"Dọn rác". Kiểm tra xem file ảnh có thực sự tồn tại và có bị hỏng hay không trước khi huấn luyện.
def filter_valid_images(image_dir, captions_dict):
    valid={} # tạo từ điển chỉ chứa ảnh sạch
    for img in captions_dict:
        path=os.path.join(image_dir, img) #nối giữa thhư mục với tên ảnh để có đường dẫn
        try:
            Image.open(path).convert("RGB")
            valid[img]=captions_dict[img] #nếu mở ảnh thành công, lấy đường dẫn của ảnh 
        except:
            continue
    return valid

#"Phẳng hóa" dữ liệu. Chuyển từ dạng từ điển phức tạp sang 2 danh sách song song để nạp vào mô hình PyTorch dễ dàng hơn.
def flatten_data(image_dir, captions_dict):
    """
    Input: 
        # Input (captions_dict):
        {
            "img1.jpg": [
                "một con chó đang chạy",
                "chó cún dễ thương chạy nhanh"
            ],
            "img2.jpg": [
                "một con mèo đang ngủ"
            ]
        }

    Output: 
        # Output 1: image_paths
        [
            "thu_muc/img1.jpg",   # Lặp lại cho câu 1
            "thu_muc/img1.jpg",   # Lặp lại cho câu 2
            "thu_muc/img2.jpg"    # Của câu 3
        ]

        # Output 2: captions
        [
            "một con chó đang chạy",
            "chó cún dễ thương chạy nhanh",
            "một con mèo đang ngủ"
        ]

    """

    image_paths, captions = [], []
    for img,caps in captions_dict.items():
        for cap in caps:
            image_paths.append(os.path.join(image_dir, img))
            captions.append(cap)
    return image_paths, captions


if __name__ == "__main__":
    import tempfile
    import sys
    from torchvision import transforms

    # Đảm bảo import được Vocabulary nếu chạy trực tiếp file
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.data.vocab import Vocabulary

    print("=== Testing flickr8k ===")
    # Tạo thư mục đóng vai trò như data_dir
    with tempfile.TemporaryDirectory() as temp_dir:
        # Giả lập file captions (1 ảnh hợp lệ, 1 ảnh lỗi không tồn tại)
        captions_file = os.path.join(temp_dir, "captions.txt")
        with open(captions_file, "w", encoding="utf-8") as f:
            f.write("image,caption\n")
            f.write("img1.jpg,a dog is running\n")
            f.write("img1.jpg,a cute dog runs\n")
            f.write("img2.jpg,a cat sleeps\n") # ảnh ảo, sẽ bị lọc

        # Tạo ảnh img1.jpg thật (Hợp lệ)
        img1_path = os.path.join(temp_dir, "img1.jpg")
        Image.new("RGB", (224, 224), color="red").save(img1_path)

        # 1. Test load_captions
        print("\n1. Testing load_captions...")
        captions_dict = load_captions(captions_file)
        print(f"   => Đọc được {len(captions_dict)} ảnh từ file:")
        for k, v in captions_dict.items():
            print(f"      - {k}: {v}")

        # 2. Test filter_valid_images
        print("\n2. Testing filter_valid_images (lọc ảnh rác)...")
        valid_dict = filter_valid_images(temp_dir, captions_dict)
        print(f"   => Sau khi lọc, còn {len(valid_dict)} ảnh tồn tại:")
        for k, v in valid_dict.items():
            print(f"      - {k}: {v}")

        # 3. Test flatten_data
        print("\n3. Testing flatten_data...")
        img_paths, flat_caps = flatten_data(temp_dir, valid_dict)
        print(f"   => Phẳng hóa thành {len(img_paths)} cặp ảnh - caption.")
        print(f"      Cặp mẫu đầu tiên: {os.path.basename(img_paths[0])} <---> '{flat_caps[0]}'")

        # 4. Test FlickrDataset
        print("\n4. Testing FlickrDataset...")
        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocab(flat_caps)
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        
        dataset = Flickr8kDataset(
            image_paths=img_paths,
            captions=flat_caps,
            vocab=vocab,
            transform=transform
        )
        
        print(f"   => Khởi tạo Dataset thành công với {len(dataset)} mẫu.")
        img_tensor, cap_tensor = dataset[0]
        print(f"   Sample 0 - Image shape: {img_tensor.shape}")
        print(f"   Sample 0 - Caption tensor: {cap_tensor}")
        
        # Thử dịch lại (Decode) mảng số ra chữ:
        reconstructed = " ".join([vocab.itos[token.item()] for token in cap_tensor])
        print(f"   Sample 0 - Decode: '{reconstructed}'")