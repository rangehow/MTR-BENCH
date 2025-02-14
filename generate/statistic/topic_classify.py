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



PROMPT='''You are tasked with classifying a series of dialogues into one of the following categories based on the content and subject matter discussed. Carefully analyze the dialogue, considering the main topics, terminology, and context. Choose the category that best fits the dialogue’s content. If none of the categories seem relevant, classify it under "Other".

Categories:

Math
Physics
Chemistry
Law
Engineering
Other
Economics
Health
Psychology
Business
Biology
Philosophy
Computer Science
Politics

Example Dialogue:
User: What’s the difference between acceleration and velocity in physics?
Assistant: Acceleration refers to the rate of change of velocity with respect to time, whereas velocity is the rate of change of position with respect to time.

Classification: Physics

Dialogue:
{history}
Classification:
'''

def history_row_query(instances,prompt,tokenizer):
    
    token_ids_list=[]
    
    for history,document_list,local_did_list in zip(instances['history'],instances['document_list'],instances['local_did']):
        
        # 处理多轮对话历史
        history_str = ""
        for i in range(0, len(history), 2):
            user_query = history[i]
            assistant_answer = history[i+1] if i+1 < len(history) else ""
            history_str += f"User: {user_query}\nAssistant: {assistant_answer}\n"

        all_text = prompt.format_map({'history':history_str})
        token_ids = tokenizer.apply_chat_template([{'role':'user','content':all_text}],add_generation_prompt=True)
        token_ids_list.append(token_ids)
    

    return {'input_ids':token_ids_list}




if __name__=="__main__":

    
    model_path='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-72B-Instruct/main'
    # model_path="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test')
    dataset = dataset.select(range(7,len(dataset),8))
    

    print("数据集列名",dataset.column_names)
    print("数据集数量",len(dataset))
    dataset = dataset.map(partial(history_row_query,prompt=PROMPT,tokenizer=tokenizer),batched=True,num_proc=32)
    print(tokenizer.decode(dataset[0]['input_ids']))
    
    sampling_params = {"temperature": 0,'max_new_tokens':256}
    llm = sgl.Engine(model_path=model_path,tp_size=4,dp_size=2 if device_count()>4 else 1,context_length=128000)
    

    print(tokenizer.decode(dataset[0]['input_ids']))
    outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)

    print("question done")
    output_text=[o['text'] for o in outputs]
    # 保存到/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp
    output_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/statistic/topic.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_text, f, ensure_ascii=False, indent=4)
