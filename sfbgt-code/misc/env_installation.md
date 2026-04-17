# sfBGT Environment Setup

## 1. Create an environment

```bash
conda create -n sfbgt python=3.10
conda activate sfbgt
pip install -r requirements.txt
```

## 2. GPU note

Install PyTorch and DGL builds that match your CUDA runtime if you want GPU training.

## 3. Run training

```bash
python main_sfbgt.py --config configs/SFBGT_CUSTOM.json
```
