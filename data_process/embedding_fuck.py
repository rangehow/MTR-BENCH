import argparse
import queue
import datasets
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets.distributed import split_dataset_by_node
from torch.multiprocessing import Process
import os
import glob
import re
import threading
import numpy as np


def parse_dataset(dataset_str):
    
    if dataset_str.endswith('.tsv'):
        # 尝试读取TSV文件
        dataset = datasets.load_dataset("csv", data_files=dataset_str, delimiter="\t",num_proc=32)['train']
    else:
        dataset = datasets.load_from_disk(dataset_str)
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Process datasets with padding and multiprocessing.")
    parser.add_argument("--model", default="stella", help="Model name or path.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size per worker.")
    parser.add_argument("--output_dir", default="./embeddings", help="Directory to save embeddings.")
    parser.add_argument("--cache_to_cpu_every", type=int, default=1000, help="Steps after which to move from GPU to CPU.")
    parser.add_argument("--cache_to_disk_every", type=int, default=10000, help="Steps after which to save to disk.")
    return parser.parse_args()

def save_embeddings(embedding_file, concatenated_embeddings):
    torch.save(concatenated_embeddings, embedding_file)
    print(f"Saved {embedding_file} ")

def get_collate_fn(tokenizer, max_length=512):
    def collate_fn(batch):
        texts = [item['text'] for item in batch]
        # 使用 tokenizer 进行分词
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        return {"texts": texts, "input_ids": encoded.input_ids, "attention_mask": encoded.attention_mask, "token_type_ids": encoded.token_type_ids}
    return collate_fn

def prefetch_generator(dataloader, device, num_streams=2, buffer_size=4):
    """
    预取生成器，提前将多个批次的数据传输到 GPU 上。

    Args:
        dataloader (DataLoader): 数据加载器。
        device (torch.device): 目标设备，例如 torch.device('cuda').
        num_streams (int): 使用的 CUDA 流数量。
        buffer_size (int): 缓冲队列的大小，即预加载的批次数量。

    Yields:
        Tuple: (input_ids_gpu, attention_mask_gpu, token_type_ids_gpu)
    """
    streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_idx = 0
    q = queue.Queue(maxsize=buffer_size)

    def worker():
        nonlocal stream_idx
        for batch in dataloader:
            stream = streams[stream_idx]
            with torch.cuda.stream(stream):
                input_ids_gpu = batch['input_ids'].to(device, non_blocking=True)
                attention_mask_gpu = batch['attention_mask'].to(device, non_blocking=True)
                token_type_ids_gpu = batch['token_type_ids'].to(device, non_blocking=True)
            # 等待当前流完成传输
            torch.cuda.current_stream(device).wait_stream(stream)
            # 将预处理后的批次放入队列
            q.put((input_ids_gpu, attention_mask_gpu, token_type_ids_gpu))
            # 切换到下一个流
            stream_idx = (stream_idx + 1) % num_streams
        # 所有数据加载完毕，放入一个标志结束的信号
        q.put(None)

    # 启动后台线程进行数据预取
    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    while True:
        batch = q.get()
        if batch is None:
            break
        yield batch

def _worker(args, rank, dataset, tokenizer):
    """
    Worker process to handle a subset of the dataset.
    """
    # 设置CUDA设备
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')
    print(f"Worker {rank} using device: {device}")

    if not torch.cuda.is_available():
        print(f"GPU {rank} is not available. Exiting.")
        return

    # 加载模型
    model_path = args.model
    print(f"Worker {rank} loading model from: {model_path}")
    model = SentenceTransformer(model_path, trust_remote_code=True, device=device,)
    model.to(device)  # 确保模型在GPU上

    # 使用统一的输出目录
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # 查找已存在的文件并确定要跳过的部分
    pattern = os.path.join(output_dir, f"embeddings_rank_{rank}_part_*.pt")
    existing_files = glob.glob(pattern)
    part_numbers = []
    for file in existing_files:
        match = re.search(r'part_(\d+)\.pt', file)
        if match:
            part_numbers.append(int(match.group(1)))
    

    if part_numbers:
        max_part = max(part_numbers)
        file_counter = max_part + 1
        num_batches_to_skip = (max_part + 1) * args.cache_to_disk_every  # 修正后的计算
        num_samples_to_skip = num_batches_to_skip * args.batch_size
        print(f"Worker {rank} found {len(part_numbers)} existing parts. Skipping {num_batches_to_skip} batches ({num_samples_to_skip} samples).")
        # 确保不超过数据集大小
        if num_samples_to_skip >= len(dataset):
            print(f"Worker {rank}: All data already processed. Exiting.")
            return
        # 跳过已处理的样本
        dataset = dataset.select(range(num_samples_to_skip, len(dataset)))
    else:
        file_counter = 0


    # 初始化DataLoader
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=8,  # 根据CPU核心数调整
        pin_memory=True,  # 启用pin_memory
        prefetch_factor=4,  # 增加预取批次数量
        collate_fn=get_collate_fn(tokenizer)
    )

    # 使用预取生成器
    data_prefetch = prefetch_generator(dataloader, device)

    batch_counter = 0
    cache_to_cpu_every = args.cache_to_cpu_every
    cache_to_disk_every = args.cache_to_disk_every

    # 一级缓存（GPU）
    gpu_cache_embeddings = []


    # 二级缓存（CPU）
    cpu_cache_embeddings = []


    for input_ids, attention_mask, token_type_ids_gpu in tqdm(data_prefetch, total=len(dataloader), desc=f"Worker {rank} processing"):
        with torch.inference_mode():
            embeddings = model.forward(dict(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids_gpu))['sentence_embedding']

        # 累积到一级缓存
        gpu_cache_embeddings.append(embeddings)
        batch_counter += 1

        # 每cache_to_cpu_every步，将一级缓存移动到二级缓存
        if batch_counter % cache_to_cpu_every == 0:
            print(f"Worker {rank}: Moving cache from GPU to CPU at batch {batch_counter}.")
            # 将GPU嵌入移动到CPU并累积到二级缓存（异步操作）
            cpu_cache_embeddings.append(torch.cat(gpu_cache_embeddings, dim=0).cpu())
            # 清空一级缓存
            gpu_cache_embeddings = []


        # 每cache_to_disk_every步，将二级缓存写入硬盘
        if batch_counter % cache_to_disk_every == 0:
            print(f"Worker {rank}: Saving cache to disk at batch {batch_counter}.")
            if cpu_cache_embeddings:
                concatenated_embeddings = torch.cat(cpu_cache_embeddings, dim=0)
                embedding_file = os.path.join(output_dir, f"embeddings_rank_{rank}_part_{file_counter}.pt")

                if os.path.exists(embedding_file):
                    print(f"Worker {rank}: Files {embedding_file} already exist. Skipping saving.")
                else:
                    save_thread = threading.Thread(target=save_embeddings, args=(embedding_file, concatenated_embeddings))
                    save_thread.start()

                # 清空二级缓存
                cpu_cache_embeddings = []
                file_counter += 1

    # 处理剩余的一级缓存
    if gpu_cache_embeddings:
        print(f"Worker {rank}: Moving remaining GPU cache to CPU.")
        cpu_cache_embeddings.append(torch.cat(gpu_cache_embeddings, dim=0).cpu())


    # 处理剩余的二级缓存
    if cpu_cache_embeddings:
        print(f"Worker {rank}: Saving remaining CPU cache to disk.")
        concatenated_embeddings = torch.cat(cpu_cache_embeddings, dim=0)
        embedding_file = os.path.join(output_dir, f"embeddings_rank_{rank}_part_{file_counter}.pt")
        texts_file = os.path.join(output_dir, f"texts_rank_{rank}_part_{file_counter}.json")

        if os.path.exists(embedding_file) and os.path.exists(texts_file):
            print(f"Worker {rank}: Files {embedding_file} and {texts_file} already exist. Skipping saving.")
        else:
            save_thread = threading.Thread(target=save_embeddings, args=(embedding_file, concatenated_embeddings))
            save_thread.start()
            save_thread.join()  # 等待保存完成

    print(f"Worker {rank} has finished processing.")



def merge_embeddings(output_dir, final_embedding_file, delete_temp_files=True):
    """
    合并所有嵌入部分文件为一个单一的嵌入文件，确保顺序与原始数据集对齐。
    可选择在合并后删除临时的部分文件。

    Args:
        output_dir (str): 存储嵌入文件的目录。
        final_embedding_file (str): 最终合并后的嵌入文件路径（.npy文件）。
        delete_temp_files (bool): 是否删除临时的部分文件。默认是True。
    """
    # 查找所有符合命名模式的嵌入文件
    embedding_files = glob.glob(os.path.join(output_dir, "embeddings_rank_*_part_*.pt"))
    
    # 过滤掉不符合规则的文件
    def is_valid_file(filename):
        return re.search(r'embeddings_rank_(\d+)_part_(\d+)\.pt', filename) is not None
    
    embedding_files = [file for file in embedding_files if is_valid_file(file)]
    
    if not embedding_files:
        print("未找到任何符合规则的嵌入部分文件。请确保嵌入文件已生成。")
        return
    
    # 提取rank和part编号，用于排序
    def extract_info(filename):
        match = re.search(r'embeddings_rank_(\d+)_part_(\d+)\.pt', filename)
        rank = int(match.group(1))
        part = int(match.group(2))
        return (rank, part)
    
    # 按rank和part排序文件
    embedding_files_sorted = sorted(embedding_files, key=lambda x: extract_info(x))
    
    print("按rank和part顺序排序后的嵌入文件列表：")
    for file in embedding_files_sorted:
        print(file)
    
    # 加载并按顺序合并嵌入
    concatenated_embeddings = []
    for file in tqdm(embedding_files_sorted, desc="合并嵌入文件"):
        embeddings = torch.load(file)  # 使用torch加载.pt文件
        embeddings = embeddings.cpu().numpy()  # 将嵌入转换为NumPy数组
        concatenated_embeddings.append(embeddings)
    
    # 将所有嵌入连接在一起
    concatenated_embeddings = np.concatenate(concatenated_embeddings, axis=0)
    
    # 保存最终合并的嵌入文件为.npy格式
    np.save(final_embedding_file, concatenated_embeddings)
    print(f"已保存合并后的嵌入文件到 {final_embedding_file}")
    
    # 删除临时文件
    if delete_temp_files:
        print("开始删除临时嵌入文件...")
        for file in embedding_files_sorted:
            try:
                os.remove(file)
                print(f"已删除临时文件: {file}")
            except Exception as e:
                print(f"删除文件 {file} 时出错: {e}")
        print("所有临时嵌入文件已删除。")
    else:
        print("未删除临时嵌入文件。")




def main():
    args = parse_args()
    print("Parsed arguments:", args)
    
    # 获取可用的 GPU 数量
    world_size = torch.cuda.device_count()
    # world_size = 1
    if world_size == 0:
        print("No GPUs available. Exiting.")
        return
    
    print(f"Number of GPUs available: {world_size}")
    model_dir = args.model
    # Initialize tokenizer and template
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print("Initialized tokenizer and template.")
    
    # Parse and preprocess the dataset
    complete_dataset = parse_dataset(args.dataset)
    

    # debug
    vector = np.load('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embeddings/final_embeddings_topiocqa.npy', mmap_mode='r')
    model = SentenceTransformer(model_dir, trust_remote_code=True, device='cuda',)
    
    vector_need_to_be_checked=model.encode(complete_dataset[10000011]['text'])
    are_close = np.allclose(vector[10000011], vector_need_to_be_checked, atol=1e-6)
    import pdb
    pdb.set_trace()








    # complete_dataset = complete_dataset.remove_columns(["date"])
    print("Parsed and preprocessed the dataset.")
    
    # 设置 world_size 为 GPU 数量
    if world_size > 1:
        dataset_list = [split_dataset_by_node(complete_dataset, rank=i, world_size=world_size) for i in range(world_size)]
    else:
        dataset_list = [complete_dataset]
    
    print("Launching worker processes...")
    
    # debug usage
    # _worker(args, 0, dataset_list[0], tokenizer)

    # Launch worker processes
    processes = []
    for rank in range(world_size):
        p = Process(target=_worker, args=(args, rank, dataset_list[rank], tokenizer))
        p.start()
        processes.append(p)
        print(f"Started process for GPU {rank}.")
    
    for p in processes:
        p.join()
        print(f"Process {p.pid} has finished.")
    
    print("All processes have finished.")
    
    # 合并所有嵌入部分文件
    final_embedding_file = os.path.join(args.output_dir, "final_embeddings")
    merge_embeddings(args.output_dir, final_embedding_file, delete_temp_files=True)  # 设置为False以保留临时文件
    
    print("Embedding merging completed.")

if __name__ == "__main__":
    main()
