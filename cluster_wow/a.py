from functools import partial
from transformers import AutoTokenizer
import numpy as np
import json
import datasets
import re
import os
from torch.cuda import device_count
import argparse

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0' # 为了加速torch编译指定的，这个版本仅限于A100，https://en.wikipedia.org/wiki/CUDA
os.environ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN']='1'


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=int)
    parser.add_argument('--end',type=int)
    return parser.parse_args()

if __name__=="__main__":

    # cluster2file=json.load(open('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/cluster_wow/cluster_to_file_mapping.json'))
    
    args=parse_args()
    start=args.start
    end=args.end
    print('start and end',start,end)



    model_path="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean')
    # dataset_list=[]
    # for i in range(start,end+1):
        
    #     dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/cluster_wow/{i}')
    #     dataset_list.append(dataset)
    
    # dataset = datasets.concatenate_datasets(dataset_list)
    # dataset = dataset.map(partial(first_row_query,prompt=PROMPT,tokenizer=tokenizer),batched=True,num_proc=32,)
    
    import pdb
    pdb.set_trace()