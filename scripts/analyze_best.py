import os
import torch
import argparse
from src.utils.config import load_config
from src.data.dataloader import get_loaders_flickr8k
from src.models import build_model
from src.evaluation.evaluator import get_detailed_results
from PIL import Image
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--env", type=str, default="local", choices=["local", "kaggle"])
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to specific checkpoint .pth file")
    args = parser.parse_args()

    config = load_config(args.config, args.env)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    data_path = config['flickr8k']['data_path']
    image_dir = os.path.join(data_path, 'images')
    captions_file = os.path.join(data_path, config["flickr8k"].get("captions_filename", "captions.txt"))

    print(f"--> [Analysis] Loading data from {data_path}...")
    loaders, vocab = get_loaders_flickr8k(
        data_dir=data_path,
        image_dir=image_dir,
        captions_file=captions_file,
        batch_size=config['data'].get('batch_size', 32),
        num_workers=config['data'].get('num_workers', 2),
    )
    _, _, test_loader = loaders

    # Build model
    model = build_model(config=config, vocab_size=len(vocab)).to(device)
    
    # Find the latest checkpoint
    root_path = config['flickr8k']['root_path']
    
    if args.checkpoint:
        latest_ckpt = args.checkpoint
        print(f"--> [Analysis] Using provided checkpoint: {latest_ckpt}")
    else:
        ckpt_dir = os.path.join(root_path, "outputs/checkpoints/transformer")
        if not os.path.exists(ckpt_dir):
            print(f"Error: Checkpoint directory not found at {ckpt_dir}. Please use --checkpoint argument.")
            return

        ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith("_best.pth")]
        if not ckpts:
            print(f"Error: No .pth checkpoints found in {ckpt_dir}")
            return
        
        ckpts.sort()
        latest_ckpt = os.path.join(ckpt_dir, ckpts[-1])
        print(f"--> [Analysis] Loading automatically found checkpoint: {latest_ckpt}")

    checkpoint = torch.load(latest_ckpt, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Analyze
    # we use beam search for better individual results
    detailed_results = get_detailed_results(model, test_loader, vocab, device, method='beam', beam_size=5)
    
    # Sort by score (BLEU-4)
    detailed_results.sort(key=lambda x: x['score'], reverse=True)
    
    top_10 = detailed_results[:10]
    bottom_10 = detailed_results[-10:]
    
    analysis_dir = os.path.join(root_path, "outputs/analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    
    def save_result(result, folder_name, idx):
        path = result['path']
        pred = result['prediction']
        refs = result['references']
        score = result['score']
        
        try:
            img = Image.open(path).convert("RGB")
            plt.figure(figsize=(10, 8))
            plt.imshow(img)
            plt.axis('off')
            
            # Show prediction and the first reference (as representative)
            # We use a wrap to handle long captions
            import textwrap
            pred_wrapped = "\n".join(textwrap.wrap(f"Predict: {pred}", width=80))
            ref_wrapped = "\n".join(textwrap.wrap(f"Reference: {refs[0]}", width=80))
            
            plt.title(f"Rank {idx} | BLEU-4: {score:.4f}\n{pred_wrapped}\n{ref_wrapped}", 
                      fontsize=10, pad=20, loc='center')
            
            save_path = os.path.join(analysis_dir, f"{folder_name}_{idx:02d}.png")
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()
            return True
        except Exception as e:
            print(f"Error saving image {path}: {e}")
            return False

    print("\n--> [Analysis] Saving Top 10 (Success)...")
    for i, res in enumerate(top_10):
        save_result(res, "success", i+1)
        
    print("--> [Analysis] Saving Bottom 10 (Failure)...")
    for i, res in enumerate(bottom_10[::-1]): # Reverse bottom to show worst first
        save_result(res, "failure", i+1)

    print(f"\nAnalysis complete! Results saved in: {analysis_dir}")

if __name__ == "__main__":
    main()
