# Image captioning using Flickr8k & IU-Xray

Project image captioning from scratch, baseline LSTM -> Transformer 

```
image-captioning-flickr8k-iu/          <-- project root 
│
├── configs/                        <--  YAML configs 
│   ├── base.yaml                   <-- defaults chung cho mọi model
│   ├── env.yaml                    <-- đường dẫn data theo môi trường (kaggle/...)
│   ├── transformer.yaml            <-- CNN + Transformer Decoder (model hiện tại)
│   └── lstm.yaml                   <-- CNN + LSTM Decoder (model đơn giản)
│
├── notebooks/                      <--  Jupyter notebooks
│   ├── 01_explore_data.ipynb       <-- EDA dataset
│   ├── 02_train_demo.ipynb         <-- Demo train nhanh
│   └── 03_evaluate.ipynb           <-- Đánh giá model sau train
│
├── outputs/                        <--  Gitignored - checkpoints, evaluations
│   └── {run_name}/
│       ├── checkpoints/
│       └── ...
│
├── scripts/                        <--  Entry points chính
│   ├── train.py                    <-- python -m scripts.train --config transformer --env kaggle
│   └── evaluate.py                 <-- python -m scripts.evaluate --config transformer --checkpoint ...
│
├── src/                            <--  Core library (tất cả code dùng chung)
│   ├── __init__.py
│   │
│   ├── data/                      
│   │   ├── __init__.py
│   │   ├── flickr8k.py             <-- Dataset class + preprocessing
│   │   ├── vocab.py                <-- Vocabulary
│   │   ├── collate.py              <-- collate_fn
│   │   └── dataloader.py           <-- get_loaders() factory
│   │
│   ├── models/                     
│   │   ├── __init__.py
│   │   ├── base.py                 <--  BaseCaptionModel abstract class
│   │   ├── encoders/               <--  Tách encoder thành module riêng
│   │   │   ├── __init__.py
│   │   │   ├── cnn_encoder.py      <-- ResNet50 encoder (từ models/cnn_encoder.py)
│   │   │   └── vit_encoder.py      <-- ViT encoder ...
│   │   ├── decoders/               <--  Tách decoder thành module riêng
│   │   │   ├── __init__.py
│   │   │   ├── lstm_decoder.py     <--  LSTM decoder
│   │   │   └── transformer_decoder.py  <-- (từ models/transformer_decoder.py)
│   │   ├── transformer.py          <-- CaptionModel CNN+Transformer (model hiện tại)
│   │   └── lstm.py                 <--  CaptionModel CNN+LSTM
│   │
│   ├── training/                  
│   │   ├── __init__.py
│   │   ├── trainer.py              <-- train_model(), train_one_epoch(), validate_one_epoch()
│   │   ├── loss.py
│   │   ├── optimizer.py            <--  get_optimizer() factory
│   │   └── scheduler.py
│   │
│   ├── evaluation/                 
│   │   ├── __init__.py
│   │   ├── evaluator.py            <-- evaluate_model(), greedy_decode()
│   │   ├── metrics.py              <-- calculate_metrics() wrapper pycocoevalcap
│   │   └── visualize.py            
│   │
│   └── utils/                      
│       ├── __init__.py
│       ├── config.py               <--  load_config() YAML loader + merge
│       ├── checkpoint.py           <--  save/load checkpoint helpers
│       ├── seed.py                 <--  set_seed()
│       ├── logger.py               <--  WandB logging wrapper
│       └── visualization.py        <--  plot helpers
│
├── tests/                          <--  Unit tests
│   ├── test_data.py
│   ├── test_models.py
│   └── test_metrics.py
│
├── pycocoevalcap/                  <-- dependency
│
├── .gitignore
├── README.md                      
└── requirements.txt
```