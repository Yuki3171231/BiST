# BiST: An Lightweight and Efficient Bi-directional Model for Spatiotemporal Prediction (Submitted to VLDB 2025)
This is the official repository of our VLDB 2025 paper. This paper introduces BiST, a lightweight yet effective **Bi**-directional **S**patio-**T**emporal prediction model based on an MLP architecture, achieving competitive predictive performance while maintaining low computational complexity and memory usage. This model effectively captures inconsistencies between label and input information to enhance performance. We propose a novel spatiotemporal decoupling module that decomposes spatiotemporal features into node-shared context features and node-specific features. We evaluate the effectiveness of the model on over a dozen datasets, including large-spatial-scale and long-period datasets. Experimental results demonstrate the effectiveness, high training efficiency, and low memory burden of our model.

<img src='Figures/main.png' alt='Main graph of BIST'>

## 1. Introduction about the datasets
### 1.1 Generating the SD and GBA sub-datasets from CA dataset
In the experiments of our paper, we used SD and GBA datasets with years from 2019 to 2021, which were generated from CA dataset, followed by [LargeST](https://github.com/liuxu77/LargeST/blob/main). For example, you can download CA dataset from the provided [link](https://www.kaggle.com/datasets/liuxu77/largest) and please place the downloaded `archive.zip` file in the `data/ca` folder and unzip the file. 

First of all, you should go through a jupyter notebook `process_ca_his.ipynb` in the folder `data/ca` to process and generate a cleaned version of the flow data. Then, please go through all the cells in the provided jupyter notebooks `generate_sd_dataset.ipynb` in the folder `data/sd` and `generate_gla_dataset.ipynb` in the folder `data/gla` respectively. Finally use the commands below to generate traffic flow data for our experiments. 
```
python data/generate_data_for_training.py --dataset sd_gba --years 2019_2020_2021
```
Moreover, you can also generate the other years of data, as well as the two additional remaining subdatasets. 

### 1.2 Generating the additional PM2.5 Knowair dataset
We implement extra experiments on [Knowair](https://github.com/shuowang-ai/PM2.5-GNN). For example, you can download Knowair dataset from the provided [link](https://drive.google.com/file/d/1R6hS5VAgjJQ_wu8i5qoLjIxY0BG7RD1L/view) and please place the downloaded `Knowair.npy` file in the `Knowair` folder and complete the files in the `Knowair/data` folder.

<br>

## 2. Environmental Requirments
The experiment requires the same environment as [LargeST](https://github.com/liuxu77/LargeST/blob/main), and need to add the libraries mentioned in the requirements in [Knowair](https://github.com/shuowang-ai/PM2.5-GNN).
