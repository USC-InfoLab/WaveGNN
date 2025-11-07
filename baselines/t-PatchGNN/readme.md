# tPatchGNN

## requirements
- wandb
- reformer_pytorch
- torch_geometric
- numpy 
- pandas
- scikit-learn

## exec baseline experiments by dataset

- MIMIC3-IHM:

```bash
cd tPatchGNN/
python run_models.py --history 48 --dataset MIMIC3-IHM --batch_size 16 --task_type binary --stride 8 --patch_size 8 --seed 42 --wandb
```

- MIMIC3-PHE:

```bash
cd tPatchGNN/
python run_models.py --history 24 --dataset MIMIC3-PHE --batch_size 16 --num_classes 25 --task_type multilabel --stride 8 --patch_size 8 --seed 42 --wandb
```

- PAM:

```bash
cd tPatchGNN/
python run_models.py --history 10 --dataset PAM --batch_size 32 --num_classes 8 --task_type multiclass --hid_dim 120 --patch_size 2 --stride 2 --seed 0 --wandb
```

- P12:

```bash
cd tPatchGNN/
python run_models.py --history 48 --dataset P12 --batch_size 16 --task_type binary --hid_dim 120 --early_stop_metric auroc --patch_size 24 --stride 24 --seed 42 --wandb
```

- P19:

```bash
cd tPatchGNN/
python run_models.py --history 1 --dataset P19 --batch_size 16 --task_type binary --hid_dim 64 --early_stop_metric auroc --patch_size 0.5 --stride 0.5 --seed 42 --wandb
```

## arguments guideline

- `history` is larger than the maximum of the timestamps
- `batch_size` shares the same unit of timestamp from dataset, not the length (kinda weird)
- `wandb` allows log records to be pushed to wandb.ai platform