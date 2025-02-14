import gc
import sys
from tqdm import tqdm
import argparse
import pickle
from collections import OrderedDict, defaultdict
import statistics
from pathlib import Path
import random
from multiprocessing import Pool
import multiprocessing
import numpy as np
import time
import json
import os
from rich.progress import track
import datasets

random.seed(0)


def jaccard_similarity(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)


def generate_ngrams(text, n=5):
    ngrams = set()
    for i in range(len(text) - n + 1):
        ngram = text[i : i + n]
        ngrams.add(ngram)
    return ngrams


def ngram_similarity(doc1, doc2, n):
    ngrams_doc1 = generate_ngrams(doc1, n)
    ngrams_doc2 = generate_ngrams(doc2, n)
    return jaccard_similarity(ngrams_doc1, ngrams_doc2)



class sort_class:
    def __init__(self, output_sort_info_subdir, output_sort_merge_dir, text_file, knn_file):
        self.text_file = text_file
        self.knn_file = knn_file

        self.num_docs = len(text_file)

        self.seen_docs = set()
        self.unseen_docs = set(range(self.num_docs))
        print(f"num docs: {self.num_docs}")

        # 拼接文档的时候还会根据相似度滤除一下
        self.doc_sim_threshold = 0.85
        self.n = 3  # n-gram
        self.output_sort_info_subdir = output_sort_info_subdir
        self.output_sort_merge_dir = output_sort_merge_dir

        self.cluster_size = 20
        self.cur_k = None
        self.filter_docs = []
        self.cluster2docs = defaultdict(list)
        self.doc2cluster = {}
        self.knns = np.load(self.knn_file, mmap_mode="r")



    def check_cluster_sizes(self):
        oversized_clusters = []
        for cluster_id, docs in self.cluster2docs.items():
            if len(docs) > self.cluster_size:
                oversized_clusters.append(cluster_id)

        if oversized_clusters:
            print(f"以下簇的大小超过了限制 {self.cluster_size}:")
            for cluster_id in oversized_clusters:
                print(f"簇 {cluster_id}: {len(self.cluster2docs[cluster_id])} 个文档")
        else:
            print(f"所有簇的大小都不超过限制 {self.cluster_size}")

    def sort(self):

        file_path = f"{self.output_sort_info_subdir}/cluster2docs.pk"
        print(file_path)
        if os.path.exists(file_path):
            print("文件已存在跳过sort")
            return

        cluster_id = 0
        # 小心这里，不算1的话第一个chunk会超出cluster_size
        cur_cluster_len = 1

        self.cur_k = self.unseen_docs.pop()
        self.cluster2docs[cluster_id].append(self.cur_k)
        self.doc2cluster[self.cur_k] = cluster_id
        self.seen_docs.add(self.cur_k)

        with tqdm(total=self.num_docs - 1,desc='sort') as pbar:
            while self.unseen_docs:
                knn = self.knns[self.cur_k, :]
                first_doc = self.output_first_doc_knn(knn)
                if (first_doc is None) or (cur_cluster_len >= self.cluster_size):

                    self.cur_k = self.unseen_docs.pop()
                    cluster_id += 1
                    cur_cluster_len = 0
                else:
                    self.cur_k = first_doc
                    self.unseen_docs.remove(self.cur_k)

                self.cluster2docs[cluster_id].append(self.cur_k)
                self.doc2cluster[self.cur_k] = cluster_id
                cur_cluster_len += 1
                self.seen_docs.add(self.cur_k)
                pbar.update(1)
        print("合并前簇的数量：", len(self.cluster2docs))
        # self.check_cluster_sizes()

        pickle.dump(
            self.cluster2docs, open(f"{self.output_sort_info_subdir}/cluster2docs.pk", "wb")
        )
        pickle.dump(self.doc2cluster, open(f"{self.output_sort_info_subdir}/doc2cluster.pk", "wb"))

    def output_first_doc_knn_not_in_the_cluster(self, knn, cluster_id):
        for k in knn:
            if k != -1:
                k_cluster = self.doc2cluster[k]

                while (
                    k_cluster != cluster_id
                ):
                    return k, k_cluster

        return None, None

    def check_all_docs_assigned(self, cluster2docs):
        all_docs = set()
        for cluster in cluster2docs.values():
            all_docs.update(cluster)

        all_docs = sorted(all_docs)

        
        print(f"文档总数是 {len(all_docs)}")

        for i, doc in enumerate(all_docs):
            if i != doc:
                print(f"缺失的文档索引：从 {i} 到 {doc - 1}")
                return False

        print("所有文档索引都已分配")
        return True

    def merge(self):
        # self.cluster2docs = pickle_load(f"{self.output_file}/cluster2docs.pk")
        # self.doc2cluster = pickle_load(f"{self.output_file}/doc2cluster.pk")
        file_path = f"{self.output_sort_info_subdir}/cluster2docs_merge.pk"
        if os.path.exists(file_path):
            print("文件已存在跳过merge")
            return

        merged_clusters_num = 0

        # 因为要在迭代中删除self.cluster2docs的东西所以得迭代copy
        for cluster, cluster_docs in tqdm(self.cluster2docs.copy().items()):
            if len(cluster_docs) < self.cluster_size:
                merged_clusters_num += 1
                # print(merged_clusters_num)
                # 如果发现了一个小于预设长度的簇，就拆散这个簇，找到里面每个元素
                for doc in cluster_docs:
                    # bp()
                    knn = self.knns[doc, :]

                    top1k, top1k_cluster = self.output_first_doc_knn_not_in_the_cluster(
                        knn, cluster
                    )
                    # bp()

                    k_cluster_docs = self.cluster2docs[top1k_cluster]

                    k_cluster_docs.insert(k_cluster_docs.index(top1k), doc)

                    # update the cluster
                    self.cluster2docs[top1k_cluster] = k_cluster_docs
                    self.doc2cluster[doc] = top1k_cluster
                del self.cluster2docs[cluster]
        print(
            f"merged_clusters_num:{merged_clusters_num},合并后簇的数量:{len(self.cluster2docs)}"
        )


        # 完备性检查
        
        self.check_all_docs_assigned(self.cluster2docs)
        pickle.dump(
            self.cluster2docs, open(f"{self.output_sort_info_subdir}/cluster2docs_merge.pk", "wb")
        )

    def output_first_doc_knn(self, knn):

        for k in knn:
            if k not in self.seen_docs and k != -1:
                return k
        return None

    
    def write_docs(self):
        sort_doc = self.cluster2list()

        # 设置每个文件的最大文档数
        max_cluster_per_file = 1000
        files = [sort_doc[i:i + max_cluster_per_file] for i in range(0, len(sort_doc), max_cluster_per_file)]

        # 确定使用的进程数
        num_processes = min(len(files), os.cpu_count())

        # 将文件列表分割成若干块，以便多进程处理
        chunks, chunk_lengths = self.divide_into_chunks(files, num_processes)

        # 计算每个进程的起始文件索引
        start_idx = np.cumsum([0] + chunk_lengths[:-1])

        # 创建参数列表
        args_list = [(chunk, start_idx[i]) for i, chunk in enumerate(chunks)]

        print(f"data ready: {len(args_list)} tasks")
        with multiprocessing.Pool(processes=num_processes) as pool:
            # 使用imap_unordered并收集每个进程返回的映射
            results = list(track(
                pool.imap_unordered(self.write_docs_wrapper, args_list),
                total=len(args_list),
            ))

        # 聚合所有进程的映射
        cluster_to_file = {}
        for mapping in results:
            cluster_to_file.update(mapping)

        # 保存映射到磁盘
        mapping_path = os.path.join(self.output_sort_merge_dir, 'cluster_to_file_mapping.json')
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump(cluster_to_file, f, ensure_ascii=False, indent=4)

        print(f"Cluster to file mapping saved to {mapping_path}")




    def divide_into_chunks(self, lst, n):
        n = min(n, len(lst))  # 确保 n 不大于列表长度
        chunks = []
        chunk_lengths = []
        for i in range(n):
            start = i * len(lst) // n
            end = (i + 1) * len(lst) // n
            chunks.append(lst[start:end])
            chunk_lengths.append(end - start)
        return chunks, chunk_lengths

    def write_docs_wrapper(self, args):
        return self.write_docs_single(*args)

    

    

    def write_docs_single(self, file_clusters_chunks, file_index):
        docs = self.text_file
        current_file_index = file_index
        cluster_to_file = {}  # 用于存储当前进程的映射
        for file_clusters in tqdm(file_clusters_chunks):
            file_data = []
            for cluster in file_clusters:
                faiss_index = cluster['faiss_index']
                doc_ids = cluster['doc_ids']
                prev_text = None
                cluster_text=[]
                doc_id_list=[]
                for doc_id in doc_ids:
                    text = docs[int(doc_id)]['text']
                    if prev_text is not None:
                        doc_sim = ngram_similarity(
                            text,
                            prev_text,
                            n=5
                        )
                        if doc_sim > 0.5:
                            continue
                    prev_text = text
                    cluster_text.append(text)
                    doc_id_list.append(doc_id)
                file_data.append({'text': cluster_text, 'did': doc_id_list, 'cid': faiss_index})

                # 记录 cluster 到文件的映射
                cluster_to_file[faiss_index] = f'{self.output_sort_merge_dir}/{current_file_index}'

            dataset = datasets.Dataset.from_list(file_data)
            dataset.save_to_disk(f'{self.output_sort_merge_dir}/{current_file_index}')
            current_file_index += 1

        return cluster_to_file
                

       



    def cluster2list(self):
        self.cluster2docs = pickle.load(
            open(f"{self.output_sort_info_subdir}/cluster2docs_merge.pk", "rb")
        )
        sort_doc = []
        for faiss_index, (cluster_id, docs) in enumerate(tqdm(self.cluster2docs.items(),desc="cluster2list")):
            sort_doc.append({
                'faiss_index': faiss_index,
                'cluster_id': cluster_id,
                'doc_ids': docs
            })
        return sort_doc




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_sort_merge_dir", type=str, default="./cluster_wow") # sort+合并数据源后的输出路径
    parser.add_argument("--output_sort_info_dir", type=str, default="./cluster_info") # 中间信息输出路径，1个part对应一个文件夹


    args = parser.parse_args()

    output_sort_info_dir = args.output_sort_info_dir
    Path(output_sort_info_dir).mkdir(parents=True, exist_ok=True)
    output_sort_merge_dir = args.output_sort_merge_dir
    Path(output_sort_merge_dir).mkdir(parents=True, exist_ok=True)


    
    neighbors='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/index/nearest_neighbors_00000.npy'
    dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean')
    

    sort_member = sort_class(
        output_sort_info_subdir=output_sort_info_dir,
        output_sort_merge_dir=output_sort_merge_dir,
        text_file=dataset, 
        knn_file=neighbors
    )
    
    sort_member.sort()
    sort_member.merge()
    sort_member.write_docs()


