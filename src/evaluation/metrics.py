import torch
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk

# Đảm bảo các resource cần thiết cho METEOR đã được tải (Kaggle thường có sẵn)
try:
    nltk.data.find('wordnet')
except LookupError:
    nltk.download('wordnet')
    nltk.download('omw-1.4')

def calculate_bleu_scores(all_preds, all_refs):
    bleu1, bleu2, bleu3, bleu4 = 0, 0, 0, 0
    chencherry = SmoothingFunction()
    
    for preds, refs in zip(all_preds, all_refs):
        bleu1 += sentence_bleu(refs, preds, weights=(1, 0, 0, 0), smoothing_function=chencherry.method1)
        bleu2 += sentence_bleu(refs, preds, weights=(0.5, 0.5, 0, 0), smoothing_function=chencherry.method1)
        bleu3 += sentence_bleu(refs, preds, weights=(0.33, 0.33, 0.33, 0), smoothing_function=chencherry.method1)
        bleu4 += sentence_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)
        
    num_samples = len(all_preds)
    return bleu1/num_samples, bleu2/num_samples, bleu3/num_samples, bleu4/num_samples

def calculate_individual_bleu(preds, refs):
    """Tính BLEU-4 cho 1 mẫu duy nhất so với danh sách refs"""
    chencherry = SmoothingFunction()
    return sentence_bleu(refs, preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=chencherry.method1)

def _lcs(x, y):
    """Hàm phụ trợ tính Longest Common Subsequence cho ROUGE-L"""
    n, m = len(x), len(y)
    table = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            if x[i] == y[j]:
                table[i + 1][j + 1] = table[i][j] + 1
            else:
                table[i + 1][j + 1] = max(table[i][j + 1], table[i + 1][j])
    return table[n][m]

def calculate_metrics(all_preds, all_refs):
    """
    Tính toán BLEU, METEOR, ROUGE-L và Accuracy.
    all_preds: list of sentences (list of tokens)
    all_refs: list of lists of sentences (list of list of tokens)
    """
    num_samples = len(all_preds)
    if num_samples == 0:
        return {}
    
    # 1. Tính BLEU
    b1, b2, b3, b4 = calculate_bleu_scores(all_preds, all_refs)
    
    # 2. Tính METEOR, Accuracy và ROUGE-L
    met_score = 0
    perfect_matches = 0
    rouge_l_score = 0
    
    for preds, refs in zip(all_preds, all_refs):
        # METEOR
        met_score += meteor_score(refs, preds)
        
        # Accuracy: Kiểm tra xem có khớp hoàn toàn với câu nào trong refs không
        if any(preds == r for r in refs):
            perfect_matches += 1
            
        # ROUGE-L (Lấy max LCS F1-score trong các câu refs)
        lcs_f1s = []
        for r in refs:
            lcs_val = _lcs(preds, r)
            precision = lcs_val / len(preds) if len(preds) > 0 else 0
            recall = lcs_val / len(r) if len(r) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            lcs_f1s.append(f1)
        rouge_l_score += max(lcs_f1s) if lcs_f1s else 0

    return {
        "BLEU-1": b1,
        "BLEU-2": b2,
        "BLEU-3": b3,
        "BLEU-4": b4,
        "METEOR": met_score / num_samples,
        "ROUGE-L": rouge_l_score / num_samples,
        "Accuracy": perfect_matches / num_samples
    }
