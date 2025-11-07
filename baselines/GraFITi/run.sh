# python train_grafiti.py \
#     --data_path /storage/datasets_public/irreg_ts/datasets/mimiciii/processed/multimodal_processed/ihm \
#     --task_name mimic3_ihm \
#     --epochs 40 \
#     --batch-size 512 \
#     --learn-rate 0.001 \
#     --latent-dim 128 \
#     --nlayers 3 \
#     --attn-head 2 \
#     --seed 0 \
#     --median_len 75 \
#     --metric_task_type binary \
#     --wandb

# python train_grafiti.py \
#     --data_path /storage/datasets_public/irreg_ts/datasets/mimiciii/processed/multimodal_processed/pheno \
#     --task_name mimic3_phe \
#     --epochs 50 \
#     --batch-size 512 \
#     --learn-rate 0.001 \
#     --latent-dim 128 \
#     --nlayers 3 \
#     --attn-head 2 \
#     --seed 0 \
#     --median_len 37 \
#     --metric_task_type multilabel \
#     --wandb

# python train_grafiti.py \
#     --data_path /storage/datasets_public/irreg_ts/datasets/P12data/ \
#     --task_name p12 \
#     --metric_task_type binary \
#     --epochs 50 \
#     --batch-size 256 \
#     --learn-rate 0.001 \
#     --latent-dim 128 \
#     --nlayers 3 \
#     --attn-head 2 \
#     --seed 0 \
#     --median_len 50 \
#     --wandb

# python train_grafiti.py \
#     --data_path /storage/datasets_public/irreg_ts/datasets/P19data/ \
#     --task_name p19 \
#     --metric_task_type binary \
#     --epochs 50 \
#     --batch-size 256 \
#     --learn-rate 0.001 \
#     --latent-dim 128 \
#     --nlayers 3 \
#     --attn-head 2 \
#     --seed 42 \
#     --median_len 50 \
#     --wandb

# python train_grafiti.py \
#     --data_path /storage/datasets_public/irreg_ts/datasets/PAMdata/ \
#     --task_name pam \
#     --metric_task_type multiclass \
#     --epochs 30 \
#     --batch-size 32 \
#     --learn-rate 0.001 \
#     --latent-dim 128 \
#     --nlayers 3 \
#     --attn-head 2 \
#     --seed 42 \
#     --median_len 600 \
    # --wandb
    # Optional PAM irregularity example:
    # --pam_apply_irregularity \
    # --pam_irregularity_rate 0.2 \
    # --pam_irregularity_type random \

# New command for PAM with random irregularity (rate=0.3)
# python train_grafiti.py \
#     --data_path /storage/datasets_public/irreg_ts/datasets/PAMdata/ \
#     --task_name pam \
#     --metric_task_type multiclass \
#     --epochs 20 \
#     --batch-size 30 \
#     --learn-rate 0.001 \
#     --latent-dim 60 \
#     --nlayers 2 \
#     --attn-head 2 \
#     --seed 42 \
#     --median_len 600 \
#     --pam_apply_irregularity \
#     --pam_irregularity_rate 0.1 \
#     --pam_irregularity_type random \
#     --wandb

# New command for PAM with fixed irregularity (rate=0.3)
python train_grafiti.py \
    --data_path /storage/datasets_public/irreg_ts/datasets/PAMdata/ \
    --task_name pam \
    --metric_task_type multiclass \
    --epochs 20 \
    --batch-size 24 \
    --learn-rate 0.001 \
    --latent-dim 80 \
    --nlayers 2 \
    --attn-head 2 \
    --seed 42 \
    --median_len 600 \
    --pam_apply_irregularity \
    --pam_irregularity_rate 0.1 \
    --pam_irregularity_type fixed \
    --wandb

