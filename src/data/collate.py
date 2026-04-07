import torch
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch):
    images = []
    captions = []

    for img, cap in batch:
        images.append(img)
        captions.append(cap)

    images = torch.stack(images)
    captions = pad_sequence(
        captions,
        batch_first=True,
        padding_value=0  # <pad>
    )

    return images, captions

if __name__ == "__main__":
    # Test batch với độ dài caption khác nhau
    # Hình ảnh giả lập có shape (3, 224, 224)
    # Caption giả lập là các tensor 1D có độ dài ngẫu nhiên
    mock_batch = [
        (torch.randn(3, 224, 224), torch.tensor([1, 4, 5, 2])),           # dài 4
        (torch.randn(3, 224, 224), torch.tensor([1, 6, 7, 8, 9, 10, 2])), # dài 7
        (torch.randn(3, 224, 224), torch.tensor([1, 11, 2]))              # dài 3
    ]
    
    batch_images, batch_captions = collate_fn(mock_batch)
    
    print("Images shape:", batch_images.shape)     # Sẽ là (3, 3, 224, 224)
    print("Captions shape:", batch_captions.shape) # Sẽ là (3, 7) do độ dài lớn nhất là 7
    print("Padded captions:\n", batch_captions)

    """
    python -m src.data.collate                                                                                                                         ─╯
        Images shape: torch.Size([3, 3, 224, 224])
        Captions shape: torch.Size([3, 7])
        Padded captions:
        tensor([[ 1,  4,  5,  2,  0,  0,  0],
                [ 1,  6,  7,  8,  9, 10,  2],
                [ 1, 11,  2,  0,  0,  0,  0]])
    """