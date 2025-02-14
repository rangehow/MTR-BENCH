import numpy as np
import faiss
import os
from tqdm import tqdm
import argparse
import concurrent.futures
import logging
import shutil

# 设置日志配置
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def parse_args():
    parser = argparse.ArgumentParser(description="Filter JSON Lines file by 'text' field.")
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing .embeds.npy files')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to the output directory')
    parser.add_argument('--topk', type=int, required=True, help='The topK value')
    parser.add_argument('--batch_size', type=int, default=2000, help='Number of files to read in parallel')
    return parser.parse_args()

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"目录 {directory_path} 已创建。")
    else:
        logging.info(f"目录 {directory_path} 已存在。")

# 获取目录下所有 .embeds.npy 文件
def get_embed_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.npy')])

# 并行加载单个文件
def load_vector(file_path):
    try:
        vectors = np.load(file_path)
        return vectors
    except Exception as e:
        logging.error(f"加载文件 {file_path} 时出错: {e}")
        return None

# 搜索最近邻并缓存结果
def search_nearest_neighbors(vectors, index, top_k, per_query_batch_size, temp_cache_dir):
    total_batches = (len(vectors) + per_query_batch_size - 1) // per_query_batch_size
    with tqdm(total=total_batches, desc="Searching nearest neighbors", leave=False) as pbar:
        for batch_num, i in enumerate(range(0, len(vectors), per_query_batch_size)):
            batch = vectors[i:i+per_query_batch_size]
            distances, indices = index.search(batch, top_k)
            # 将结果保存为 numpy 数组
            batch_results = indices.astype(np.int32)
            # 定义临时文件路径
            cache_file = os.path.join(temp_cache_dir, f"nearest_neighbors_batch_{batch_num:05d}.npy")
            np.save(cache_file, batch_results)
            pbar.update(1)

def initialize_faiss_index(embedding_dim: int) -> faiss.Index:
    index = faiss.IndexFlatL2(embedding_dim)
    return index

def make_vres_vdev(ngpus, gpu_resources, i0=0, i1=-1):
    "Return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.Int32Vector()
    if i1 == -1:
        i1 = ngpus
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev

def move_index_to_gpu(index: faiss.Index, ngpus, gpu_resources) -> faiss.Index:
    """
    将FAISS索引移动到GPU（如果可用）
    """
    if ngpus > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        vres, vdev = make_vres_vdev(ngpus, gpu_resources)
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        logging.info("FAISS索引已成功移动到GPU。")
        return gpu_index
    else:
        logging.warning("没有可用的GPU，使用CPU索引。")
        return index

# 合并临时缓存文件为一个最终文件
def merge_temp_cache_files(temp_cache_dir, final_output_file):
    cache_files = sorted([
        os.path.join(temp_cache_dir, f) for f in os.listdir(temp_cache_dir)
        if f.startswith("nearest_neighbors_batch_") and f.endswith(".npy")
    ])
    
    if not cache_files:
        logging.error("没有找到任何缓存文件来合并。")
        return
    
    # 计算总行数
    total_rows = 0
    sample = None
    for f in cache_files:
        data = np.load(f)  # 此时不用 mmap_mode='r'
        total_rows += data.shape[0]
        if sample is None:
            sample = data
    if sample is None:
        logging.error("所有缓存文件均为空或加载失败。")
        return
    total_cols = sample.shape[1]
    
    # 在内存中创建一个大的 array
    # 注意：如果 total_rows 非常大，且内存有限，可能会内存不够
    merged = np.zeros((total_rows, total_cols), dtype=np.int32)
    
    current_index = 0
    for f in cache_files:
        data = np.load(f)  # 读入
        rows = data.shape[0]
        merged[current_index:current_index+rows] = data
        current_index += rows
        logging.info(f"已合并文件 {f}")
    
    # 将完整的数据用 np.save 写到文件 final_output_file
    # 建议带上 .npy 后缀
    np.save(final_output_file, merged)
    logging.info(f"所有缓存文件已成功合并到 {final_output_file}，且总行数为{total_rows}")

    # 删除临时缓存文件
    shutil.rmtree(temp_cache_dir)
    logging.info(f"已删除临时缓存目录 {temp_cache_dir}。")


# 处理缓冲区中的向量
def process_vectors(file_index, vectors, ngpus, gpu_resources, args, top_k, per_query_batch_size, embedding_dim, output_dir):
    # 创建一个专属的临时缓存目录
    temp_cache_dir = os.path.join(output_dir, f"temp_cache_{file_index}")
    create_directory(temp_cache_dir)
    
    # 创建FAISS索引
    index = initialize_faiss_index(embedding_dim)
    index = move_index_to_gpu(index, ngpus, gpu_resources)
    logging.info(f"当前处理第 {file_index} 个分片，包含 {vectors.shape[0]} 个向量。")

    index.add(vectors)

    logging.info(f"索引创建完成。总共索引了 {index.ntotal} 个向量。")

    # 搜索最近邻，并将结果缓存到本地
    search_nearest_neighbors(
        vectors,
        index,
        top_k=top_k,
        per_query_batch_size=per_query_batch_size,
        temp_cache_dir=temp_cache_dir
    )

    # 合并临时缓存文件为一个最终的邻居文件
    final_output_file = os.path.join(output_dir, f"nearest_neighbors_{int(file_index):05d}.npy")
    merge_temp_cache_files(temp_cache_dir, final_output_file)

    # 可选：检查结果完整性（如有需要）
    # check_completeness(vectors, index, top_k)

def main():
    args = parse_args()

    # 设置参数
    EMBEDDING_DIM = 1024
    PER_QUERY_BATCH_SIZE = 2097152   # 根据需要调整2097152
    TOP_K = args.topk
    MAX_VECTORS_PER_FILE = int(2e7)
    BATCH_SIZE = args.batch_size  # 每批处理的文件数量
    MAX_WORKERS = os.cpu_count()  # 并行读取的线程数

    # 初始化 FAISS GPU 资源
    ngpus = faiss.get_num_gpus()
    logging.info(f"Number of GPUs: {ngpus}")
    
    gpu_resources = []
    for i in range(ngpus):
        res = faiss.StandardGpuResources()
        gpu_resources.append(res)

    # 创建输出目录
    create_directory(args.output_dir)

    # 获取所有嵌入文件
    embed_files = get_embed_files(args.input_dir)
    
    if not embed_files:
        raise ValueError(f"No .embeds.npy files found in directory {args.input_dir}")

    # 初始化变量
    buffer = []
    count = 0
    total_lines = 0

    # 分批处理文件
    for i in tqdm(range(0, len(embed_files), BATCH_SIZE), desc="Processing file batches"):
        batch_files = embed_files[i:i+BATCH_SIZE]
        
        # 并行加载文件，保持顺序
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # 使用executor.map保持结果顺序
            vectors_list = list(executor.map(load_vector, batch_files))
        
        for file_path, vectors in zip(batch_files, vectors_list):
            if vectors is not None:
                logging.info(f"加载文件 {file_path} 成功，包含 {vectors.shape[0]} 个向量。")
                buffer.append(vectors)
                total_lines += vectors.shape[0]

                # 检查缓冲区是否超过阈值
                current_size = sum(v.shape[0] for v in buffer)
                while current_size >= MAX_VECTORS_PER_FILE:
                    # 将缓冲区中的向量拼接
                    all_vectors = np.concatenate(buffer, axis=0)
                    batch_vectors = all_vectors[:MAX_VECTORS_PER_FILE]
                    logging.info(f"正在处理第 {count} 个分片，包含 {batch_vectors.shape[0]} 个向量。")
                    process_vectors(
                        file_index=count, 
                        vectors=batch_vectors, 
                        ngpus=ngpus, 
                        gpu_resources=gpu_resources, 
                        args=args, 
                        top_k=TOP_K, 
                        per_query_batch_size=PER_QUERY_BATCH_SIZE, 
                        embedding_dim=EMBEDDING_DIM,
                        output_dir=args.output_dir
                    )
                    count += 1
                    # 更新缓冲区
                    remaining_vectors = all_vectors[MAX_VECTORS_PER_FILE:]
                    if remaining_vectors.size > 0:
                        buffer = [remaining_vectors]
                    else:
                        buffer = []
                    current_size = sum(v.shape[0] for v in buffer)
                    logging.info(f"当前buffer共有 {current_size} 行！！！")
                    logging.info(f"所有emb加起来共计 {total_lines} 行")

    # 处理剩余的向量
    if buffer:
        all_vectors = np.concatenate(buffer, axis=0)
        logging.info(f"正在处理最后一个分片，包含 {all_vectors.shape[0]} 个向量。")
        process_vectors(
            file_index=count, 
            vectors=all_vectors, 
            ngpus=ngpus, 
            gpu_resources=gpu_resources, 
            args=args, 
            top_k=TOP_K, 
            per_query_batch_size=PER_QUERY_BATCH_SIZE, 
            embedding_dim=EMBEDDING_DIM,
            output_dir=args.output_dir
        )
        count += 1

    logging.info(f"所有文件处理完成，共生成 {count} 个邻居结果文件。")

if __name__ == "__main__":
    ngpus = faiss.get_num_gpus()
    logging.info(f"Number of GPUs: {ngpus}")
    main()
