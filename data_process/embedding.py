import logging
import os
import numpy as np
import datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
from sentence_transformers import LoggingHandler, SentenceTransformer
from sentence_transformers.models import Pooling,Transformer
import argparse

logging.basicConfig(
    format="%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO, handlers=[LoggingHandler()]
)

def parse_dataset(dataset_str):
    if dataset_str.endswith('.tsv'):
        # 尝试读取TSV文件
        dataset = datasets.load_dataset("csv", data_files=dataset_str, delimiter="\t", num_proc=32)['train']
    else:
        dataset = datasets.load_from_disk(dataset_str)
    return dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Process datasets with padding and multiprocessing.")
    parser.add_argument("--model", default="stella", help="Model name or path.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset.")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size per worker.")
    parser.add_argument("--output_dir", default="./embeddings", help="Directory to save embeddings.")
    parser.add_argument("--model_name")
    return parser.parse_args()

if __name__ == "__main__":
    # Set params

    args = parse_args()


    # 获取倒数第二个部分
    model_name = args.model_name
    print(model_name)
    # 载入数据集
    dataset = parse_dataset(args.dataset)

    if 'qwen' in model_name:
        model = SentenceTransformer(args.model, trust_remote_code=True)
    else:
        model1 = Transformer(args.model, model_args={'trust_remote_code':True})
        pooling_model = Pooling(model1.get_word_embedding_dimension(),pooling_mode='cls')
        model = SentenceTransformer(modules=[model1, pooling_model])
        # model = SentenceTransformer(args.model, trust_remote_code=True)


    # vector = np.load('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embeddings/all_embeddings.npy', mmap_mode='r')
    
    
    # vector_need_to_be_checked=model.encode(dataset[10000011]['text'])
    # are_close = np.allclose(vector[10000011], vector_need_to_be_checked, atol=1e-6)
    # import pdb
    # pdb.set_trace()




    # 启动多进程池
    pool = model.start_multi_process_pool()


    # 设置DataLoader
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=8192*16 ,
        num_workers=8,  # 根据CPU核心数调整
        pin_memory=True,  # 启用pin_memory
        prefetch_factor=4,  # 增加预取批次数量
    )

    # Ensure the output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # 用于存储所有批次的嵌入
    all_embeddings = []

    for i, batch in enumerate(tqdm(dataloader)):
        if "title" in batch:
            sentences = [f"{t}\n{x}" for t, x in zip(batch["title"], batch["text"])]
        else:
            sentences = batch["text"]



        batch_emb = model.encode_multi_process(sentences, pool, batch_size=2048 if 'qwen' not in model_name else 64,chunk_size=2048 if 'qwen' not in model_name else 64)

        if np.any(np.all(batch_emb == 0, axis=1)):
            print(f"Warning: All-zero embeddings found in batch {i+1}.")
            import pdb
            pdb.set_trace()


        print(f"Embeddings computed for batch {i+1}. Shape: {batch_emb.shape}")
        
        all_embeddings.append(batch_emb)


    all_embeddings = np.vstack(all_embeddings)
    print(f"Total embeddings shape: {all_embeddings.shape}")

    

    output_file = os.path.join(args.output_dir, f"all_embeddings_{model_name}.npy")
    np.save(output_file, all_embeddings)

    
    print(f"All embeddings saved at {output_file}")
    
    if np.any(np.all(all_embeddings == 0, axis=1)):
        print("Warning: All-zero embeddings found in final output.")

    model.stop_multi_process_pool(pool)

    print("All embeddings have been processed and saved.")






# vector = np.load('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embeddings/final_embeddings_topiocqa.npy', mmap_mode='r')
    
    
# vector_need_to_be_checked=model.encode(dataset[10000011]['text'])
# are_close = np.allclose(vector[10000011], vector_need_to_be_checked, atol=1e-6)
# import pdb
# pdb.set_trace()