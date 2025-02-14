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

PROMPT='''You need to determine whether humans can answer the Current Query based on the given document. Based on the likelihood of the answer being found in the document, assign a score from 1 to 3, where:

- 1: Cannot answer the Current Query
- 2: Can partially answer the Current Query
- 3: Can answer the Current Query

You only need to return the score, without any explanation.
'''


def history_row_query(instances, prompt, tokenizer):
    token_ids_list = []
    
    for messages, g_ctx in zip(instances['messages'], instances['ground_truth_ctx']):
        prompt_str = prompt
        # 处理多轮对话历史
        history_str = ""
        for message in messages[:-1]:
            if message['role'] == 'user':
                history_str += f"User: {message['content']}\n"
            else:
                history_str += f"Assistant: {message['content']}\n"
            
        prompt_str += f"\nDocument:\n{g_ctx['ctx']}"
        if history_str != "":
            prompt_str += f"\nDialogue History:\n{history_str}"
        prompt_str += f"\nCurrent Query:\n{messages[-1]['content']}\n"
        prompt_str += f"Score:"

        token_ids = tokenizer.apply_chat_template([{'role':'user','content':prompt_str}], add_generation_prompt=True)
        token_ids_list.append(token_ids)
    
    return {'input_ids': token_ids_list}


if __name__ == "__main__":
    # 使用 argparse 获取 model_dir
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Path to the model directory")
    args = parser.parse_args()
    
    model_path = args.model_dir
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-chatrag-format')
    
    print("数据集列名", dataset.column_names)
    print("数据集数量", len(dataset))

    dataset = dataset.map(partial(history_row_query, prompt=PROMPT, tokenizer=tokenizer), batched=True, num_proc=32)
    print(tokenizer.decode(dataset[1]['input_ids']))
    
    sampling_params = {"temperature": 0, 'max_new_tokens': 256}
    llm = sgl.Engine(model_path=model_path, tp_size=4, context_length=8192 * 2)
    
    print(tokenizer.decode(dataset[0]['input_ids']))
    outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)

    print("question done")
    output_text = [o['text'] for o in outputs]

    # 修改保存文件名为 score-模型名字
    model_name = os.path.basename(os.path.dirname(model_path))
    save_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/statistic/score-{model_name}.json'
    
    json.dump(output_text, open(save_path, 'w'), indent=4)
