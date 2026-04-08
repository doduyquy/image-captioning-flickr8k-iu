import torch
from tqdm import tqdm
from .metrics import calculate_metrics

@torch.no_grad()
def evaluate_model(model, dataloader, vocab, device, method='greedy', beam_size=5, max_len=25):
    """
    Duyệt qua dataloader, sinh caption và tính điểm BLEU.
    """
    model.eval()
    all_preds = []
    all_refs = []
    
    print(f"--> [Evaluation] Đang đánh giá trên {len(dataloader.dataset)} mẫu ({method})...")
    
    # Ở Flickr8k, mỗi ảnh thường có 5 câu mô tả.
    # Dataset của ta hiện tại trả về (img, 1_cap) lặp lại cho mỗi câu.
    # Để tính BLEU chuẩn, ta cần gom các câu mô tả của cùng 1 ảnh lại.
    
    for images, captions in tqdm(dataloader):
        images = images.to(device)
        
        # Sinh caption cho batch này
        # Để đơn giản và chính xác, ta lặp qua từng ảnh trong batch
        for i in range(images.size(0)):
            img = images[i].unsqueeze(0)
            pred_tokens = model.generate(img, vocab, method=method, beam_size=beam_size, max_len=max_len, device=device)
            
            # Decode tokens -> words
            pred_words = [vocab.itos[t.item()] for t in pred_tokens if vocab.itos[t.item()] not in ["<start>", "<end>", "<pad>"]]
            all_preds.append(pred_words)
            
            # Đối với references: Một ảnh có thể xuất hiện nhiều lần trong loader với các caption khác nhau.
            # Trong một phiên bản chuyên sâu, ta nên dùng Grouped DataLoader. 
            # Ở đây ta lấy caption hiện tại làm reference (tối thiểu 1 câu).
            ref_words = [vocab.itos[t.item()] for t in captions[i] if vocab.itos[t.item()] not in ["<start>", "<end>", "<pad>"]]
            all_refs.append([ref_words]) # BLEU yêu cầu list of lists cho refs
            
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
    
    with torch.no_grad():
        for images, captions in dataloader:
            if samples_shown >= num_samples:
                break
                
            img = images[0].unsqueeze(0).to(device)
            pred_tokens = model.generate(img, vocab, method=method, device=device)
            
            pred_sentence = " ".join([vocab.itos[t.item()] for t in pred_tokens if vocab.itos[t.item()] not in ["<start>", "<end>", "<pad>"]])
            ref_sentence = " ".join([vocab.itos[t.item()] for t in captions[0] if vocab.itos[t.item()] not in ["<start>", "<end>", "<pad>"]])
            
            print(f"\nVí dụ {samples_shown + 1}:")
            print(f"  Gốc: {ref_sentence}")
            print(f"  Dự đoán: {pred_sentence}")
            
            show_prediction(images[0], pred_sentence, [ref_sentence])
            samples_shown += 1

