# scHNTL
single-cell RNA-seq data clustering augmented by high-order neighbors and triplet loss

## Configuration environment

Python --- 3.6.4

Tensorflow --- 1.12.0

Keras --- 2.1.5

Numpy --- 1.19.5

Scipy --- 1.5.2

Pandas --- 1.1.5

Sklearn --- 0.24.2

## Using procedure


### Get datasets
All the original datasets ([Biase](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE57249), [Klein](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE65525), [Romanov](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE74672), [Björklund](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE70580), [PBMC](https://support.10xgenomics.com/single-cell-gene-expression/datasets/1.1.0/pbmc6k), [Sun.2,Sun.3](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE128066), [Qs_Diaphragm,Qx_Bladder,Adam](https://figshare.com/articles/software/scBGEDA/19657911), [Baron](https://github.com/LiShenghao813/AttentionAE-sc/tree/main/Data), [Brown2](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE137710)) can be downloaded. 

### Pre-process
If the datasets need, before clustering, low-quality cells and genes can be filtered.
Take dataset Bisae for example. The original expression matrix `ori_data.tsv` of dataset Biase is downloaded and put into `/data/Biase`. 
By running the following command: 
```Bash
python preprocess.py Biase
```
the pre-processed expression matrix `data.tsv` is produced under `/data/Biase`.

### Run scHNTL
To use scHNTL, you need two parameters, `dataset_str` and `n_clusters`, where `dataset_str` is the name of dataset and `n_clusters` is the number of clusters.
Then run the following command:
```Bash
python scHNTL.py dataset_str n_clusters
```
For dataset `Biase`, you can run the following command:
```Bash
python scHNTL.py Biase 3
```
For your own dataset named `Dataset_X`, you can first create a new folder `/data/Dataset_X`, and put the expression matrix file `data.tsv` into `/data/Dataset_X`, then run scHNTL on it.

### Outputs
You can obtain the predicted clustering result `pred_DatasetX.txt` and the learned cell embeddings `hidden_DatasetX.tsv` under the folder `/result`.
