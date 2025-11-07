# task select from ['mor', 'los', 'decom', 'wbm', 'vent', 'vaso', 'active', 'physio']
# --dp_flag: use DataParallel

# small scale MIMIC-III

# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --d_model 32 --batch 32 --n_head 1 --n_layers 2 --d_k 8 --d_v 8 --lr 1e-3 --epoch 2 --patience 5 \
#     --log /path/to/log/ --save_path /path/to/save/ \
#     --task 'mor' --seed 0 --dp_flag --warp_num '0_1' 


# large scale  MIMIC-III
# for large sacle, --load_in_batch is required

# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --d_model 32 --batch 16 --n_head 1 --n_layers 2 --d_k 8 --d_v 8 --lr 1e-3 --epoch 40 --patience 5 \
#     --log /path/to/log/ \
#     --save_path /path/to/save/ \
#     --task 'wbm' --seed 0 --dp_flag --warp_num '0_1' --load_in_batch


# PhysioNet (median len. 72)

# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --batch 32 --lr 1e-3 --epoch 50 --patience 5 \
#     --log /path/to/log/ --save_path /path/to/save/ \
#     --task 'physio' --seed 0 --warp_num '0_0.2_1' \
#     --batch_size 32 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
#     --dropout 0.0 --n_head 1 --n_layers 3 


# Human Activity (median len. 50)
# for human activity, perform per time point classification

# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --batch 64 --lr 1e-3 --epoch 50 --patience 5 \
#     --log /path/to/log/ \
#     --save_path /path/to/save/ \
#     --task 'active' --seed 0 --warp_num '1.4_0.2_1' 

# PAM
# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --batch 32 --lr 1e-3 --epoch 50 --patience 5 \
#     --log log/ --save_path save/ \
#     --task 'pam' \
#     --batch_size 32 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
#     --dropout 0.0 --n_head 1 --n_layers 3 \
#     --seed 0 \
#     --wandb # use wandb to log the results online

# P12
# python Main_warp.py \
#     --data_path /path/to/datasets/ \
#     --batch 32 --lr 1e-3 --epoch 50 --patience 5 \
#     --log log/ --save_path save/ \
#     --task 'p12' \
#     --batch_size 32 --d_inner_hid 64 --d_k 8 --d_model 64 --d_v 8 \
#     --dropout 0.0 --n_head 1 --n_layers 3 \
#     --seed 42 \
#     --wandb # use wandb to log the results online

# P19
python Main_warp.py \
    --data_path /path/to/datasets/ \
    --batch 32 --lr 1e-3 --epoch 50 --patience 5 \
    --log log/ --save_path save/ \
    --task 'p19' \
    --batch_size 128 --d_inner_hid 32 --d_k 4 --d_model 32 --d_v 4 \
    --dropout 0.0 --n_head 1 --n_layers 3 \
    --seed 37 \
    --wandb # use wandb to log the results online

# MIMIC-III IHM
# python Main_warp.py \
#     --data_path /storage/datasets_public/irreg_ts/datasets/mimiciii/processed/multimodal_processed/ihm \
#     --task 'mimic3_ihm' \
#     --batch_size 128 --lr 1e-3 --epoch 50 --patience 7 \
#     --log log/ --save_path save/ \
#     --d_model 64 --n_head 1 --n_layers 3 \
#     --seed 0 \
#     --wandb

# MIMIC-III PHE
# python Main_warp.py \
#     --data_path /storage/datasets_public/irreg_ts/datasets/mimiciii/processed/multimodal_processed/pheno \
#     --task 'mimic3_phe' \
#     --batch_size 128 --lr 1e-3 --epoch 50 --patience 7 \
#     --log log/ --save_path save/ \
#     --d_model 64 --n_head 1 --n_layers 3 \
#     --seed 0 \
#     --wandb