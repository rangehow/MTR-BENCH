from functools import partial
import sglang as sgl
from transformers import AutoTokenizer
import numpy as np
import json
import datasets
import re
import os
from torch.cuda import device_count
import argparse
from tqdm import tqdm

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0' # 为了加速torch编译指定的，这个版本仅限于A100，https://en.wikipedia.org/wiki/CUDA
os.environ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN']='1'

PROMPT='''Rewrite the following search query to be more effective for information retrieval. Focus on clarity, specificity, and using keywords that will likely return the most relevant results. Output rewrite query only.
'''


def history_row_query(instances,prompt,tokenizer):
    
    token_ids_list=[]
    
    for messages in instances['messages']:
        prompt_str=prompt
        # 处理多轮对话历史
        history_str = ""
        for message in messages[:-1]:
            
            if message['role']=='user':
                history_str+=f"User: {message['content']}\n"
            else:
                history_str+=f"Assistant: {message['content']}\n"
            

        if history_str!="":
            prompt_str+=f"\nDialogue History:\n{history_str}"
        prompt_str+=f"\nOriginal Query:\n{messages[-1]['content']}\n"
        prompt_str+=f"Rewrite Query:"

       
        # print(prompt_str)
        
        
        token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt_str}],add_generation_prompt=True)
        token_ids_list.append(token_ids)
    

    return {'input_ids':token_ids_list}



if __name__=="__main__":

    
    # model_path='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-7B-Instruct/main'
    model_path = '/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/meta-llama/Llama-3.1-8B-Instruct/main'
    # model_path="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    # model_path='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-72B-Instruct/main'
    # model_path='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/google/gemma-2-27b-it/main'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    

    dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired-chatrag-format-hard')
    
    print("数据集列名",dataset.column_names)
    print("数据集数量",len(dataset))

    dataset = dataset.map(partial(history_row_query,prompt=PROMPT,tokenizer=tokenizer),batched=True,num_proc=32,)
    sampling_params = {"temperature": 0,'max_new_tokens':512}
    llm = sgl.Engine(model_path=model_path,tp_size=1,dp_size=4,context_length=8192)
    
    print(tokenizer.decode(dataset[0]['input_ids']))
    outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)

    print("question done")
    output_text=[o['text'] for o in outputs]
    
    
    dataset = dataset.add_column(name='rewrite_query',column=output_text)
    
    dataset = dataset.remove_columns('input_ids')

    dataset.save_to_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired-chatrag-format-hard-rewrite-llama8b')

