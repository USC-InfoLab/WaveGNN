## 📢 Publication

This work was accepted at **IEEE BigData 2025 (Macau, China)** and received **Best Student Paper Runner-up**.

**Paper (arXiv):** https://arxiv.org/abs/2412.10621  
**Award:** https://conferences.cis.um.edu.mo/ieeebigdata2025/awards.html (photo: [link](https://github.com/user-attachments/assets/b5a464f0-a444-45ff-be2c-67a0e4ec1afa))

**Citation:**
Arash Hajisafi, Maria Despoina Siampou, Bita Azarijoo, Zhen Xiong, and Cyrus Shahabi.  
"**WaveGNN: Integrating Graph Neural Networks and Transformers for Decay-Aware Classification of Irregular Clinical Time-Series.**"  
*IEEE International Conference on Big Data (IEEE BigData 2025).*


# setting up the environment

```
conda create -n wave-gnn python=3.9.19
conda activate wave-gnn
pip install -r requirements.txt
```

# running

```
python run.py --dataset `dataset_name`
```
Dataset options are: P12, P19, PAM, MIMIC3-PHE for phenotype classification and MIMIC3-IHM for in hospital mortality prediction.
Each dataset is accompanied by its own config file.


# datasets used in this experimental evaluation:

We used [Raindrop's](https://github.com/mims-harvard/Raindrop) preprocessing for [P12](https://figshare.com/articles/dataset/P12_dataset_for_Raindrop/19514341/1?file=34683085), [P19](https://figshare.com/articles/dataset/P19_dataset_for_Raindrop/19514338/1?file=34683070), and [PAM](https://figshare.com/articles/dataset/PAM_dataset_for_Raindrop/19514347/1?file=34683103). 
However, [MIMIC-III](https://physionet.org/content/mimiciii/1.4/) is a restricted-access dataset, thus, we are unable to share preprocessed dataset directly. If you would like to access the data you can follow the followings steps:

- Complete all the requisites through MIMIC-III's [official page](https://physionet.org/content/mimiciii/1.4/#files) to obtain the raw data.
- Clone [MIMIC-III-Benchmark](https://github.com/YerevaNN/mimic3-benchmarks/) and do the preprocessing for `in-hospital-mortality` and `phenotyping` tasks to get preprocessed data. For `phenotyping`, you can restrict period length to 24 by adding a line in [`readers.py`](https://github.com/YerevaNN/mimic3-benchmarks/blob/ea0314c7cbd369f62e2237ace6f683740f867e3a/mimic3benchmark/readers.py#L257) to get the first 24 hours of admission data and make it comparable with our setting.
- Clone [UTDE](https://github.com/XZhang97666/MultimodalMIMIC?tab=readme-ov-file) repository and run [`preprocessing.py`](https://github.com/XZhang97666/MultimodalMIMIC/blob/main/preprocessing.py) to get normalized data. 


# hyperparameters

The description of hyperparameters are as follows:
- *main_path*: The parent directory in which all datasets reside.
- *data_path*: The file/directory in which the dataset exists.
- *label_path*: The file that contains labels for P12, P19, and PAM.
- *splits_path*: The directory that contains different splits for train, validation, and test sets. For fair comparison we utilized Raindrop's splits. Therefore, it exists only for P12, P19, and PAM.
- *save_dir*: Directory to save WaveGNN's results.
- *period_length*: Length of ICU admission in hours that we used in our analysis (e.g. 48 for MIMIC3-IHM and 24 for MIMIC3-PHE).
- *batch_size*: Size of each batch.
- *gradient_accumulation_step*: Number of steps to accumulate gradients before applying optimizer. This parameter is required to fit large data in memory.
- *epochs*: Number of epochs.
- *window_size*: The length of multivariate time series used. Each window has its own timestamp which shows when a multivariate time series record ocurred.
- *lr*: Initial learning rate.
- *dropout*: Dropout rate.
- *patience*: Number of steps to take before terminating training if there is no improvement in terms of an evaluation metric (e.g. AUROC, AUPRC, F1) on validation set.
- *weight_decay*: Weight decay.
- *scheduler*: Whether to use scheduler for adjusting learning rate.
- *random_seed*: Random seed for reproducing results.
- *observation_dim*: Dimension of a single observation value of a sensor in a specific timestamp (default to 1).
- *num_attention_heads*: Number of attention heads uses in `MultiheadAttention` module.
- *irregularity*: Whether to add irregularity in the data. Dafault to `False`.
- *irregularity_rate*: Percentage of sensors to remove observations from (default to 0).
- *irregularity_type*: How to remove sensors if `irregularity_rate` is specified. Choices are from `['fixed', 'random']`. `fixed` means most informative `irregularity_rate` sensors are removed whereas `random` means sensors are removed at random.
- *n_classes*: Number of classes used for classification. In binary classification, it is set to `1` since we use Sigmoid activation function.
- *time_encoding*: Type of time encoding to use. Choices are among `['absolute_transformer', 'relative_t2v']`. `absolute_transformer` applies original transformer's positional encoding while `relative_t2v` shifts timestamps to a reference point, calculates a relative timestamp similar to GRU-D, and applies Time2Vec for encoding relative timestamps.
- *device*: Device to run WaveGNN on. Choices are `['cuda', 'cpu']`.
- *use_wandb*: Whether to use Wandb to log hyperparameters and track the experiment on Wandb servers. You need to have Wandb API key set in your system to use this feature (defaults to false).
- *n_runs* : How many times to run WaveGNN.
- *n_splits*: Number of different dataset splits. 
