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

PROMPT='''Rewrite the last query in a series of historical dialogues by replacing nouns with appropriate pronouns or omitting them where suitable according to historical dialogues. Ensure that the rewritten query can be understood based on the dialogue history. If you cannot rewrite it appropriately, keep the original query. Do not provide reasons, only return the rewritten query.

Examples:

1. Coreference (Pronoun Replacement):

Dialogue History:

Query: Where is John?
Response: He went to the shop.
Query: What did John buy?

Rewritten Query:
What did he buy?

2. Ellipsis (Omission):

Dialogue History:

Query: Where do you want to travel?
Response: I want to travel to France.
Query: Why do you want to travel to France?

Rewritten Query:
Why?
'''


def history_row_query(instances,prompt,tokenizer):
    
    token_ids_list=[]
    
    for messages in instances['messages']:
        prompt_str=prompt
        # 处理多轮对话历史
        history_str = ""
        for message in messages:
            
            if message['role']=='user':
                history_str+=f"Query: {message['content']}\n"
            else:
                history_str+=f"Response: {message['content']}\n"
            

        
        prompt_str+=f"\nReal User Input:\n\nDialogue History:\n\n{history_str}"
        prompt_str+=f"Rewritten Query:"

        
        
        token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt_str}],add_generation_prompt=True)
        token_ids_list.append(token_ids)
    

    return {'input_ids':token_ids_list}

def change(instance,i,output_text):
    origin=instance['messages']
    if i%8!=0 and i%4!=0:
        
        origin[-1]['content']=output_text[i]
    return {'messages':origin}
        

    

if __name__=="__main__":

    dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-chatrag-format')
    
    print("数据集列名",dataset.column_names)
    print("数据集数量",len(dataset))
    # model_path="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    model_path='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-72B-Instruct/main'

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = dataset.map(partial(history_row_query,prompt=PROMPT,tokenizer=tokenizer),batched=True,num_proc=1,)
    dataset = dataset.shuffle(seed=42)
    dataset = dataset.select(range(20000))
    print(tokenizer.decode(dataset[1]['input_ids']))
    
    
    

    sampling_params = {"temperature": 0,'max_new_tokens':256}
    llm = sgl.Engine(model_path=model_path,tp_size=4,context_length=12800)
    
    print(tokenizer.decode(dataset[0]['input_ids']))
    outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)

    print("question done")
    output_text=[o['text'] for o in outputs]
    json.dump(output_text,open('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/statistic/tmp.json','w'))
    # output_text=json.load(open('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/statistic/tmp.json','r'))
    dataset = dataset.map(partial(change,output_text=output_text),num_proc=32,with_indices=True)
    

    dataset.save_to_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-chatrag-format-hard')
    
    
    

