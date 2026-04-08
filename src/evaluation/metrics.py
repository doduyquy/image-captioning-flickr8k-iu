import torch
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_scores(all_preds, all_refs):
    """
    all_preds: list of sentences (list of tokens)
    all_refs: list of lists of sentences (list of list of tokens)
    """
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    chencherry = SmoothingFunction()
    
    for preds, refs in zip(all_preds, all_refs):
        bleu1 += sentence_bleu(refs, preds, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
        bleu2 += sentence_bleu(refs, preds, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
        bleu3 += sentence_bleu(refs, preds, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
        bleu4 += sentence_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        
    num_samples = len(all_preds)
    if num_samples == 0:
        return 0, 0, 0, 0
        
    return bleu1/num_samples, bleu2/num_samples, bleu3/num_samples, bleu4/num_samples

def calculate_metrics(all_preds, all_refs):
    """
    Wrapper để tính toán tất cả các metrics.
    Hiện tại mới chỉ có BLEU. Có thể mở rộng thêm ROUGE, CIDEr sau.
    """
    b1, b2, b3, b4 = calculate_bleu_scores(all_preds, all_refs)
    
    return {
        "BLEU-1": b1,
        "BLEU-2": b2,
        "BLEU-3": b3,
        "BLEU-4": b4
    }
