import os
import datasets
import shutil
from tqdm import tqdm


def correct(instances):
    history = []
    for query, answer in zip(instances['query'], instances['answer']):
        history.append([query, answer])
    return {'history': history}

# 原始数据集目录
source_dir = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage1_old/'
# 目标数据集目录
target_dir = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage1/'

# 确保目标目录存在
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历原始目录下的所有子目录或文件
for item in tqdm(os.listdir(source_dir)):
    item_path = os.path.join(source_dir, item)
    
    # 确保处理的是目录（假设每个数据集存储在一个目录中）
    if os.path.isdir(item_path):
        print(f"Processing dataset: {item}")
        
        # 加载数据集
        dataset = datasets.load_from_disk(item_path)
        # 用来记录已经处理过的 cid
        seen_cids = set()

        # 定义过滤函数，保留第一次出现的 cid
        def filter_fn(example):
            # 如果 cid 还没有出现过，就保留当前条目
            if example['cid'] not in seen_cids:
                seen_cids.add(example['cid'])
                return True  # 保留该条目
            return False  # 过滤掉重复的条目


        # 过滤数据集，只保留第一次出现的 cid,这个要是设置超过1会变成傻逼，多进程操作set会炸开。
        dataset = dataset.filter(filter_fn,num_proc=1)
        assert len(dataset)==1000,len(dataset)
        # 应用自定义函数进行处理
        dataset = dataset.map(correct, num_proc=64, batched=True)
        
        # 移除多余的列
        dataset = dataset.remove_columns(['query', 'answer'])
        
        # 保存到目标目录
        target_path = os.path.join(target_dir, item)
        dataset.save_to_disk(target_path)
        print(f"Saved processed dataset to: {target_path}")
