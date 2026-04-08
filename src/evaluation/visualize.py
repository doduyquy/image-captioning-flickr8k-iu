import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

def show_prediction(image_tensor, pred_caption, ref_captions=None, save_path=None):
    """
    Hiển thị ảnh cùng với câu dự đoán và (tùy chọn) các câu gốc.
    image_tensor: [3, H, W] tensor sau khi transform
    pred_caption: Chuỗi ký tự dự đoán
    ref_captions: Danh sách chuỗi gốc
    """
    # Unnormalize/Rescale ảnh về dạng [0, 1] để pyplot vẽ được
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    # Nếu ảnh có chuẩn hóa Mean/Std (ImageNet), cần đảo ngược (inverse) 
    # Nhưng ở đây ta cứ kẹp 0-1 cho đơn giản.
    img = (img - img.min()) / (img.max() - img.min()) 
    
    plt.figure(figsize=(10, 8))
    plt.imshow(img)
    plt.axis('off')
    
    title = f"Dự đoán: {pred_caption}"
    if ref_captions:
        title += f"\nTham chiếu: {ref_captions[0]}" # Hiện 1 câu mẫu
    
    plt.title(title, fontsize=12)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def plot_loss(train_losses, val_losses, save_path=None):
    """ Vẽ đồ thị loss sau khi train xong """
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss')
    
    if save_path:
        plt.savefig(save_path)
    plt.show()
