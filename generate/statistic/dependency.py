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

PROMPT='''Given a series of historical dialogues and a current query, determine if the current query contains any coreference (anaphora) or ellipsis phenomena.

If coreference is present, explain the coreference and return 1.
If ellipsis is present, explain the ellipsis and return 2.
If neither coreference nor ellipsis is present, return 0.
Example 1 (Coreference):

Dialogue History:

A: "John went to the store."
B: "Did he buy anything?"
A: "Yes, he bought some apples."
Current Query: "What did he buy?"

Reasoning: The pronoun "he" in the current query refers to "John" from the previous sentences, indicating coreference.

Response: 1

Example 2 (Ellipsis):

Dialogue History:

A: "I want to go to the park."
B: "I want to go too."
Current Query: "Where do you want to go?"

Reasoning: The phrase "to the park" is omitted in the current query, but it is understood from the prior dialogue, indicating ellipsis.

Response: 2
'''


def history_row_query(instances,prompt,tokenizer):
    
    token_ids_list=[]
    
    for messages in instances['messages']:
        prompt_str=prompt
        # 处理多轮对话历史
        history_str = ""
        for message in messages[:-1]:
            
            if message['role']=='user':
                history_str+=f"A: {message['content']}\n"
            else:
                history_str+=f"B: {message['content']}\n"
            

        if history_str!="":
            prompt_str+=f"\nDialogue History:\n{history_str}"
        prompt_str+=f"\nCurrent Query: {messages[-1]['content']}\n"
        prompt_str+=f"Reasoning:"

       
        
        
        token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt_str}],add_generation_prompt=True)
        token_ids_list.append(token_ids)
    

    return {'input_ids':token_ids_list}



if __name__=="__main__":

    

    model_path='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-72B-Instruct/main'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired-chatrag-format')
    
    print("数据集列名",dataset.column_names)
    print("数据集数量",len(dataset))

    dataset = dataset.map(partial(history_row_query,prompt=PROMPT,tokenizer=tokenizer),batched=True,num_proc=32,)

    sampling_params = {"temperature": 0,'presence_penalty':0.1,'max_new_tokens':512}
    llm = sgl.Engine(model_path=model_path,tp_size=4,context_length=128000)
    
    print(tokenizer.decode(dataset[0]['input_ids']))
    outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)

    print("question done")
    output_text=[o['text'] for o in outputs]
    json.dump(output_text,open('/tmp/dependency.json','w'),indent=4)
    labels = []
    for text in output_text:
        match = re.search(r'Response: (\d)', text)
        if match:
            labels.append(int(match.group(1)))
        else:
            labels.append(None)  # 如果没有找到匹配，加入None
    
    # 统计0, 1, 2的数量
    label_counts = {0: labels.count(0), 1: labels.count(1), 2: labels.count(2)}
    
    # 输出占比
    total = len(labels)
    label_proportions = {k: v / total for k, v in label_counts.items()}
    
    # 打印结果
    print(f"占比统计: {label_proportions}")
    
    # 保存结果到文件
    result_file = "output_results.json"
    with open(result_file, 'w') as f:
        json.dump({
            "label_counts": label_counts,
            "label_proportions": label_proportions
        }, f, indent=4)

    print(f"结果已保存到 {result_file}")
    
    

