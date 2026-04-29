import torch
from tqdm import tqdm
from .metrics import calculate_metrics, calculate_individual_bleu

from collections import defaultdict

@torch.no_grad()
def evaluate_model(model, dataloader, vocab, device, method='greedy', beam_size=5, max_len=25):
    """
    Duyệt qua dataloader, gom nhóm references theo ảnh, sinh 1 caption duy nhất cho mỗi ảnh 
    và tính điểm BLEU, METEOR, ROUGE-L chuẩn.
    """
    model.eval()
    
    image_to_preds = {} # path -> list of words
    image_to_refs = defaultdict(list) # path -> list of list of words
    
    print(f"--> [Evaluation] Đang gom nhóm và đánh giá trên {len(dataloader.dataset)} mẫu ({method})...")
    
    special_tokens = {"<start>", "<end>", "<pad>", "<unk>"}
    
    for images, captions, paths in tqdm(dataloader):
        # 1. Thu thập References và Sinh dữ liệu nếu chưa có
        for i in range(len(paths)):
            path = paths[i]
            
            # Chuyển caption tensor -> list of words
            ref_words = [vocab.itos[t.item()] for t in captions[i] if vocab.itos[t.item()] not in special_tokens]
            image_to_refs[path].append(ref_words)
            
            # Nếu chưa sinh caption cho ảnh này, thực hiện sinh
            if path not in image_to_preds:
                img = images[i].unsqueeze(0).to(device)
                pred_tokens = model.generate(img, vocab, method=method, beam_size=beam_size, max_len=max_len, device=device)
                
                # Decode tokens -> words
                pred_words = [vocab.itos[t.item()] for t in pred_tokens if vocab.itos[t.item()] not in special_tokens]
                image_to_preds[path] = pred_words
                
    # 2. Chuẩn bị dữ liệu cho calculate_metrics
    # all_preds: list of [word1, word2, ...]
    # all_refs: list of [ [[ref1_w1, ...]], [[ref2_w1, ...]], ... ]
    
    unique_paths = list(image_to_preds.keys())
    all_preds = [image_to_preds[p] for p in unique_paths]
    all_refs = [image_to_refs[p] for p in unique_paths]
    
    print(f"--> [Evaluation] Hoàn thành sinh caption cho {len(all_preds)} ảnh duy nhất. Đang tính metrics...")

    # Tính toán các metrics
    metrics_results = calculate_metrics(all_preds, all_refs)
    
    for name, val in metrics_results.items():
        print(f"   {name}: {val:.4f}")
        
    return metrics_results

def evaluate_and_show(model, dataloader, vocab, device, method='greedy', num_samples=3):
    """
    Vừa đánh giá vừa hiển thị một vài mẫu trực quan.
    """
    from .visualize import show_prediction
    
    model.eval()
    samples_shown = 0
    special_tokens = {"<start>", "<end>", "<pad>", "<unk>"}
    
    with torch.no_grad():
        for images, captions, paths in dataloader:
            if samples_shown >= num_samples:
                break
                
            img = images[0].unsqueeze(0).to(device)
            pred_tokens = model.generate(img, vocab, method=method, device=device)
            
            pred_sentence = " ".join([vocab.itos[t.item()] for t in pred_tokens if vocab.itos[t.item()] not in special_tokens])
            # Hiển thị câu ref đầu tiên của ảnh này trong batch
            ref_sentence = " ".join([vocab.itos[t.item()] for t in captions[0] if vocab.itos[t.item()] not in special_tokens])
            
            print(f"\nVí dụ {samples_shown + 1}:")
            print(f"  Gốc (mẫu): {ref_sentence}")
            print(f"  Dự đoán: {pred_sentence}")
            
            # images[0] có thể có shape [2, 3, H, W] (IU X-Ray: frontal + lateral)
            # hoặc [3, H, W] (Flickr8k: single image). Luôn lấy ảnh đầu tiên.
            img_to_show = images[0]
            if img_to_show.dim() == 4:
                img_to_show = img_to_show[0]  # Lấy frontal view
            show_prediction(img_to_show, pred_sentence, [ref_sentence])
            samples_shown += 1

@torch.no_grad()
def get_detailed_results(model, dataloader, vocab, device, method='greedy', beam_size=5, max_len=25):
    """
    Tương tự evaluate_model nhưng trả về danh sách chi tiết từng ảnh 
    kèm điểm số để phân tích thành công/thất bại.
    """
    model.eval()
    image_to_preds = {}
    image_to_refs = defaultdict(list)
    special_tokens = {"<start>", "<end>", "<pad>", "<unk>"}
    
    print(f"--> [Analysis] Đang trích xuất kết quả chi tiết...")
    
    for images, captions, paths in tqdm(dataloader):
        for i in range(len(paths)):
            path = paths[i]
            ref_words = [vocab.itos[t.item()] for t in captions[i] if vocab.itos[t.item()] not in special_tokens]
            image_to_refs[path].append(ref_words)
            
            if path not in image_to_preds:
                img = images[i].unsqueeze(0).to(device)
                pred_tokens = model.generate(img, vocab, method=method, beam_size=beam_size, max_len=max_len, device=device)
                pred_words = [vocab.itos[t.item()] for t in pred_tokens if vocab.itos[t.item()] not in special_tokens]
                image_to_preds[path] = pred_words
                
    detailed_results = []
    for path in image_to_preds:
        preds = image_to_preds[path]
        refs = image_to_refs[path]
        score = calculate_individual_bleu(preds, refs)
        
        detailed_results.append({
            "path": path,
            "prediction": " ".join(preds),
            "references": [" ".join(r) for r in refs],
            "score": score
        })
        
    return detailed_results

