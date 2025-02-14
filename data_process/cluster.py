import torch
import faiss
import numpy as np
import json
import os
from datasets import load_from_disk
from tqdm import tqdm

# 1. 从本地存储读取 x
x_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/embeddings/final_embeddings.pt'
print(f"加载 x 数据：{x_path}")
x = torch.load(x_path).float().numpy()  # 转换为 NumPy 数组，确保数据类型为 float32
print(f"数据加载完成，形状：{x.shape}")

# 2. 加载数据集
dataset_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/dedup'
print(f"加载数据集：{dataset_path}")
dataset = load_from_disk(dataset_path)

# 确保 x 的行数与数据集的行数一致
assert x.shape[0] == len(dataset), "x 的行数与数据集的行数不一致！"

# 3. 设置参数
ncentroids = max(1, x.shape[0] // 200)
niter = 20
verbose = True
d = x.shape[1]
print(f"初始化 KMeans，簇数：{ncentroids}，维度：{d}，迭代次数：{niter}")
kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose,gpu=True)

# 4. 训练 KMeans
print(f"开始训练 KMeans...")
kmeans.train(x)
print("KMeans 训练完成。")

# 5. 分批查询并合并结果
batch_size = 1000  # 每批处理的大小
cluster_ids = []  # 保存每个元素的簇号

print(f"开始分批查询，共 {len(x)} 条数据，每批处理 {batch_size} 条。")
for start_idx in tqdm(range(0, x.shape[0], batch_size), desc="查询进度", ncols=100):
    end_idx = min(start_idx + batch_size, x.shape[0])
    batch_x = x[start_idx:end_idx]
    D, I = kmeans.index.search(batch_x, 1)  # 对每一批进行查询
    cluster_ids.extend(I.flatten())  # 将结果展平并保存
print("所有数据的查询完成。")

# 创建保存文件夹
output_dir = './cluster'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"文件夹 '{output_dir}' 已创建/存在。")

# 保存簇号到 .npy 文件
cluster_ids_path = os.path.join(output_dir, 'cluster_ids.npy')
np.save(cluster_ids_path, np.array(cluster_ids))
print(f"簇号数据已保存到 {cluster_ids_path}。")

# 6. 根据聚类结果将数据分成独立的 JSON 文件
clusters = {}
print(f"开始根据聚类结果分组数据，共有 {len(cluster_ids)} 个元素。")
for idx, cluster_id in tqdm(enumerate(cluster_ids), desc="分配数据", total=len(cluster_ids), ncols=100):
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(dataset[idx])  # 将相应的数据添加到簇中

# 将每个簇保存为独立的 JSON 文件
print(f"开始保存每个簇的数据为独立的 JSON 文件。")
for cluster_id, items in clusters.items():
    cluster_file_path = os.path.join(output_dir, f'cluster_{cluster_id}.json')
    with open(cluster_file_path, 'w') as f:
        json.dump(items, f)
    print(f"簇 {cluster_id} 的数据已保存为 {cluster_file_path}。")

print("聚类过程完成，所有结果已保存。")
