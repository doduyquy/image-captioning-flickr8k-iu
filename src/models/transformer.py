import torch
import torch.nn as nn
from .base import BaseCaptionModel
from .CNN.swin_encoder import SwinEncoder
from .CNN.cnn_encoder_fusion import CNNEncoderFusion
from .CNN.hybrid_encoder import HybridEncoder
from .encoders.transformer_encoder import TransformerEncoder
from .decoders.transformer_decoder import TransformerDecoder

class TransformerCaptionModel(BaseCaptionModel):
    """
    Model ghép nối CNN -> Spatial Transformer (Encoder) -> Transformer Decoder.
    """
    def __init__(self, config, vocab_size):
        super().__init__()
        
        embed_dim = config['model'].get('embed_dim', 512)
        num_heads = config['model'].get('num_heads', 8)
        ff_dim = config['model'].get('ff_dim', 2048)
        max_len = config['model'].get('max_len', 100) # Flickr8k thường max 40-50, 100 là an toàn
        
        # 1. Trích xuất đặc trưng hạ tầng (CNN)
        # self.cnn_encoder = CNNEncoder(embed_dim=embed_dim)
        # self.cnn_encoder_fusion = CNNEncoderFusion(embed_dim=embed_dim)
        # self.cnn_encoder_swin = SwinEncoder(embed_dim=embed_dim)    
        self.encoder = HybridEncoder(embed_dim=embed_dim)
        # 2. Xử lý không gian bằng Transformer Encoder
        self.spatial_encoder = TransformerEncoder(
            embed_dim=embed_dim, 
            num_heads=num_heads
        )
        
        # 3. Sinh văn bản (Decoder)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            max_len=max_len
        )

    def _encode(self, images):
        """
        Encode ảnh, hỗ trợ cả 2 mode:
          - Single-image [B, 3, H, W]   → Flickr8k
          - Dual-image   [B, 2, 3, H, W] → IU_Xray (average features của 2 ảnh)

        Returns: features [B, S, embed_dim] sau spatial encoder
        """
        if images.dim() == 5:  # [B, 2, 3, H, W]
            B, N, C, H, W = images.shape
            # Reshape → [B*N, 3, H, W] → encode → average
            flat = images.view(B * N, C, H, W)
            feats = self.encoder(flat)                          # [B*N, S, D]
            feats = feats.view(B, N, *feats.shape[1:])         # [B, N, S, D]
            feats = feats.mean(dim=1)                          # [B, S, D]
        else:                  # [B, 3, H, W]
            feats = self.encoder(images)                       # [B, S, D]

        return self.spatial_encoder(feats)                     # [B, S, D]

    def forward(self, images, captions):
        """
        images: [B, 3, 224, 224]  hoặc  [B, 2, 3, 224, 224]
        captions: [B, T]
        """
        features = self._encode(images)      # [B, S, D]
        logits = self.decoder(captions, features)
        return logits

    @torch.no_grad()
    def generate(self, images, vocab, method='greedy', **kwargs):
        """
        Giao diện chính để sinh caption từ ảnh.
        Args:
            images: Tensor của ảnh [B, 3, 224, 224]
            vocab: Đối tượng vocabulary
            method: 'greedy' hoặc 'beam'
            **kwargs: Tham số phụ như beam_size, max_len, v.v.
        """
        self.eval()
        if method == 'greedy':
            return self._greedy_decode(images, vocab, **kwargs)
        elif method == 'beam':
            return self._beam_search_decode(images, vocab, **kwargs)
        else:
            raise ValueError(f"Không hỗ trợ phương pháp decoding: {method}")

    def _greedy_decode(self, images, vocab, max_len=25, device='cpu', **kwargs):
        """
        Giải mã tham lam (Greedy Search): Ở mỗi bước chọn từ có xác suất cao nhất.
        Hỗ trợ cả single [B,3,H,W] và dual [B,2,3,H,W] ảnh.
        """
        images = images.to(device)
        features = self._encode(images)      # [B, S, D]

        start_token = vocab.stoi["<start>"]
        captions = torch.tensor([[start_token]]).to(device)

        for _ in range(max_len):
            logits = self.decoder(captions, features)
            next_token = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            captions = torch.cat([captions, next_token], dim=1)
            if next_token.item() == vocab.stoi["<end>"]:
                break

        return captions.squeeze(0)

    def _beam_search_decode(self, images, vocab, beam_size=5, max_len=25, device='cpu', **kwargs):
        """
        Giải mã Beam Search: Duy trì K (beam_size) giả thuyết tốt nhất ở mỗi bước.
        Hỗ trợ cả single [B,3,H,W] và dual [B,2,3,H,W] ảnh.
        """
        images = images.to(device)
        start_token = vocab.stoi["<start>"]
        end_token = vocab.stoi["<end>"]

        features = self._encode(images)   # [B, S, D]

        # 2. Khởi tạo Beam
        # Một beam item: (chuỗi các token, điểm xác suất tích lũy)
        beams = [([start_token], 0.0)]
        completed_beams = []
        
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == end_token:
                    completed_beams.append((seq, score))
                    continue
                
                # Predict next token
                input_tensor = torch.tensor([seq]).to(device)
                logits = self.decoder(input_tensor, features)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1).squeeze(0)
                
                # Lấy top k ứng viên tiếp theo
                topk_probs, topk_indices = torch.topk(log_probs, beam_size)
                
                for i in range(beam_size):
                    next_token = topk_indices[i].item()
                    next_score = score + topk_probs[i].item()
                    new_beams.append((seq + [next_token], next_score))
            
            # Chỉ giữ lại top K (beam_size) các nhánh tốt nhất từ tất cả các nhánh con
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_size]
            
            # Nếu tất cả các nhánh trong beam đều đã kết thúc
            if all(seq[-1] == end_token for seq, score in beams):
                break
        
        # Gom kết quả
        final_results = completed_beams + beams
        final_results.sort(key=lambda x: x[1], reverse=True)
        
        # Trả về chuỗi tốt nhất dưới dạng Tensor
        best_seq = final_results[0][0]
        return torch.tensor(best_seq).to(device)