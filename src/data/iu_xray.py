import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image


class IUXrayDataset(Dataset):
    """
    Dataset cho IU X-Ray (R2Gen-style annotation.json).

    Mỗi sample trả về:
        - images: tensor stack cả 2 ảnh [2, 3, H, W] (cả frontal và lateral)
        - caption: tensor numericalized report
        - image_path: đường dẫn ảnh đầu tiên (dùng làm key trong evaluation)
    
    Lưu ý: Việc xác định frontal/lateral không định rõ trong dataset,
    nên ta encode cả 2 ảnh và average features trong model (giống R2Gen).
    """

    def __init__(self, image_paths_pairs, captions, vocab, transform=None):
        """
        Args:
            image_paths_pairs: list of [path_img1, path_img2] cho mỗi sample
            captions: list of report strings tương ứng
            vocab: Vocabulary object
            transform: torchvision transforms
        """
        self.image_paths_pairs = image_paths_pairs
        self.captions = captions
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.image_paths_pairs)

    def __getitem__(self, idx):
        paths = self.image_paths_pairs[idx]   # [path1, path2]

        # Load và transform từng ảnh
        imgs = []
        for p in paths:
            img = Image.open(p).convert("RGB")
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        # Stack thành [2, 3, H, W]
        images = torch.stack(imgs, dim=0)

        caption = self.captions[idx]
        tokens = self.vocab.numericalize(caption)
        numericalized = [self.vocab.stoi["<start>"]]
        numericalized += tokens
        numericalized.append(self.vocab.stoi["<end>"])

        # path key cho evaluation: dùng ảnh đầu tiên
        return images, torch.tensor(numericalized), paths[0]


def load_iu_xray_annotation(annotation_file):
    """
    Đọc file annotation.json theo R2Gen-style.

    Format JSON:
        [
            {
                "id": "1",
                "image": ["frontal.png", "lateral.png"],
                "report": "The heart is ...",
                "split": "train"
            },
            ...
        ]

    Args:
        annotation_file (str): Đường dẫn đến annotation.json

    Returns:
        dict: { "train": [...], "val": [...], "test": [...] }
              Mỗi phần tử là 1 dict gốc từ JSON
    """
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Không tìm thấy annotation file: {annotation_file}")

    with open(annotation_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    split_data = {"train": [], "val": [], "test": []}
    for item in data:
        split = item.get("split", "train")
        if split in split_data:
            split_data[split].append(item)
        else:
            # Fallback nếu split value không hợp lệ
            split_data["train"].append(item)

    counts = {k: len(v) for k, v in split_data.items()}
    print(f"   IU_Xray annotation loaded — Train: {counts['train']}, Val: {counts['val']}, Test: {counts['test']}")
    return split_data


def flatten_iu_xray(image_dir, items):
    """
    Chuyển list annotation items thành 2 danh sách song song:
    image_paths_pairs và captions.

    Mỗi sample giữ CẢ 2 ảnh và lưu dưới dạng [path1, path2].
    Không cần biết cái nào là frontal/lateral.
    Sample bị bỏ qua nếu thiếu report hoặc không đủ 2 ảnh tồn tại.

    Args:
        image_dir (str): Thư mục chứa các file ảnh
        items (list): List annotation items [{image: [p1, p2], report: str}]

    Returns:
        image_paths_pairs (list): Danh sách [path1, path2] cho mỗi sample
        captions (list): Danh sách report tương ứng
    """
    image_paths_pairs, captions = [], []
    skipped = 0

    for item in items:
        img_names = item.get("image", [])
        report = item.get("report", "").lower().strip()

        # Bỏ qua sample thiếu dữ liệu
        if not report:
            skipped += 1
            continue

        # Lấy đường dẫn cho tất cả ảnh có trong item
        valid_paths = []
        for name in img_names:
            p = os.path.join(image_dir, name)
            if os.path.exists(p):
                valid_paths.append(p)

        if len(valid_paths) == 0:
            # Không có ảnh nào tồn tại
            skipped += 1
            continue
        elif len(valid_paths) == 1:
            # Chỉ có 1 ảnh: dùng lại nó cho cả 2 slot
            valid_paths = valid_paths * 2
        else:
            # Có ≥ 2 ảnh: chỉ lấy 2 cái đầu
            valid_paths = valid_paths[:2]

        image_paths_pairs.append(valid_paths)
        captions.append(report)

    if skipped > 0:
        print(f"   [IU_Xray] Đã bỏ qua {skipped} samples (thiếu ảnh hoặc report rỗng)")

    return image_paths_pairs, captions


# -----------------------------------------------------------------------
# Quick test — chạy bằng: python -m src.data.iu_xray
# -----------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import tempfile
    from torchvision import transforms

    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from src.data.vocab import Vocabulary

    print("=== Testing IUXrayDataset ===\n")

    with tempfile.TemporaryDirectory() as temp_dir:
        # --- Tạo ảnh giả lập ---
        img1_path = os.path.join(temp_dir, "frontal_1.png")
        img2_path = os.path.join(temp_dir, "frontal_2.png")
        Image.new("RGB", (224, 224), color="gray").save(img1_path)
        Image.new("RGB", (224, 224), color="white").save(img2_path)

        # --- Tạo annotation.json giả lập ---
        annotation_file = os.path.join(temp_dir, "annotation.json")
        fake_data = [
            {
                "id": "1",
                "image": ["frontal_1.png", "lateral_1.png"],
                "report": "the heart size is normal no pleural effusion",
                "split": "train"
            },
            {
                "id": "2",
                "image": ["frontal_2.png", "lateral_2.png"],
                "report": "mild cardiomegaly is noted lungs are clear",
                "split": "val"
            },
            {
                "id": "3",
                "image": ["missing_frontal.png"],  # ảnh không tồn tại
                "report": "this should be skipped",
                "split": "test"
            }
        ]
        with open(annotation_file, "w", encoding="utf-8") as f:
            json.dump(fake_data, f)

        # 1. Test load_iu_xray_annotation
        print("1. Testing load_iu_xray_annotation...")
        split_data = load_iu_xray_annotation(annotation_file)
        for split, items in split_data.items():
            print(f"   {split}: {len(items)} items")

        # 2. Test flatten_iu_xray
        print("\n2. Testing flatten_iu_xray (train split)...")
        img_paths, caps = flatten_iu_xray(temp_dir, split_data["train"])
        print(f"   => {len(img_paths)} cặp hợp lệ")
        if img_paths:
            print(f"   Cặp đầu: {os.path.basename(img_paths[0])} <-> '{caps[0]}'")

        # 3. Test IUXrayDataset
        print("\n3. Testing IUXrayDataset...")
        all_paths = img_paths
        all_caps = caps

        vocab = Vocabulary(freq_threshold=1)
        vocab.build_vocab(all_caps)

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        dataset = IUXrayDataset(all_paths, all_caps, vocab, transform=transform)
        print(f"   Dataset size: {len(dataset)}")

        img_tensor, cap_tensor, path = dataset[0]
        print(f"   Image shape: {img_tensor.shape}")
        print(f"   Caption tensor: {cap_tensor}")
        decoded = " ".join([vocab.itos[t.item()] for t in cap_tensor])
        print(f"   Decoded: '{decoded}'")
        print("\n=== All tests passed! ===")
