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

PROMPT='''Rewrite the User’s part in the following conversation, by replacing the nouns in questions with appropriate pronouns or by making suitable omissions. Delete expressions like “according to the document” and similar ones. 


## Original Dialogue
{history}

## Rewrite Dialogue
'''



# Here comes an example:
# ## Original Dialogue
# User: According to the document, What was the name of the band Molland formed after leaving Badfinger in 1974?
# Assistant: Molland formed a band called Natural Gas after leaving Badfinger in late 1974. The band included Jerry Shirley, who was formerly of Humble Pie. They released their self-titled album in 1976 and toured successfully with Peter Frampton in 1977. However, the band disbanded late in 1977 due to a lack of organization.
# User: What was the main reason for the band Natural Gas's demise late in 1977?
# Assistant: The main reason for the band's demise was a general lack of organization.
# User: Who did Molland and Evans recruit to promote the "Airwaves" album after Harck was fired from the band?
# Assistant: Molland and Evans recruited Tony Kaye, the former keyboardist of Yes, to promote the "Airwaves" album. They also brought in Peter Clarke on drums from Stealers Wheel.
# User: What was the peak position of "Love Is Gonna Come at Last" on the Billboard charts?
# Assistant: The single "Love Is Gonna Come at Last" from the album "Airwaves" reached a peak position of number 69 on the Billboard charts.
# User: What was the highest chart position reached by the "Airwaves" album on the US charts?
# Assistant: -The \"Airwaves\" album reached a peak position of number 125 on the US charts.

# ## Rewrite Dialogue
# User: What was the name of the band Molland formed after leaving Badfinger in 1974?
# Assistant: Molland formed a band called Natural Gas after leaving Badfinger in late 1974. The band included Jerry Shirley, who was formerly of Humble Pie. They released their self-titled album in 1976 and toured successfully with Peter Frampton in 1977. However, the band disbanded late in 1977 due to a lack of organization.
# User: What was the main reason for their demise late in 1977?
# Assistant: The main reason for the band's demise was a general lack of organization.
# User: Who did Molland and Evans recruit to promote the "Airwaves" album after Harck was fired from the band?
# Assistant: Molland and Evans recruited Tony Kaye, the former keyboardist of Yes, to promote the "Airwaves" album. They also brought in Peter Clarke on drums from Stealers Wheel.
# User: What was the peak position of "Love Is Gonna Come at Last" on the Billboard charts?
# Assistant: The single "Love Is Gonna Come at Last" from the album "Airwaves" reached a peak position of number 69 on the Billboard charts.
# User: What was the highest position reached by it on the US charts?
# Assistant: The album reached a peak position of number 125 on the US charts.

# Rewrite the User’s part in the following conversation, by replacing the nouns in questions with appropriate pronouns or by making suitable omissions. Avoid expressions like “according to the document” and similar ones. For example, in the following sample conversation, simplify "Natural Gas Band" to just "Band" and refer to the album "Airwaves" as "it." Keep the number of conversation rounds at 5.

def history_row_query(instances,prompt,tokenizer):
    
    token_ids_list=[]
    
    for history in instances['history']:
        
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




def split_dialogue(text):
    # 使用正则表达式分割文本，捕获User/Assistant标识
    parts = re.split(r'(User|Assistant):\s*', text)
    result = []
    current_role = None
    
    for part in parts:
        if part in ('User', 'Assistant'):
            # 遇到角色标识时记录当前角色
            current_role = part
        elif current_role is not None:
            # 遇到内容时添加并重置角色标识
            result.append(part.strip())
            current_role = None
    
    return result






def add_answer(instances):

    history_list = []
    for history, answer in zip(instances['history'], instances['answer']):
        history_list.append(history+[answer])

    return {'history': history_list}

def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--start',type=int)
    parser.add_argument('--end',type=int)
    parser.add_argument('--stage',type=int)
    return parser.parse_args()

if __name__=="__main__":

    args=parse_args()
    start=args.start
    end=args.end
    stage=args.stage
    print('start and end',start,end)
    # if os.path.exists(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage1/s{start}e{end}'):
    #     print("数据集已存在，退出")
    #     exit()

    model_path='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-72B-Instruct/main'
    # model_path="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage{stage-1}/s{start}e{end}')
  
    print("数据集列名",dataset.column_names)
    print("数据集数量",len(dataset))
    dataset = dataset.map(partial(history_row_query,prompt=PROMPT,tokenizer=tokenizer),batched=True,num_proc=32,)
    sampling_params = {"temperature": 0.7, "top_p": 0.95,'max_new_tokens':10000}
    llm = sgl.Engine(model_path=model_path,tp_size=4,dp_size=2 if device_count()>4 else 1,context_length=128000)
    # if not os.path.exists(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_questions_s{start}d{end}.json'):


    print(tokenizer.decode(dataset[0]['input_ids']))
    outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)
    
    print("question done")
    output_text=[o['text'] for o in outputs]
    
    # 保存到/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp
    output_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_questions_s{start}d{end}.json'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_text, f, ensure_ascii=False, indent=4)

    # else:
    #     print("复用了已生成的问题数据集")
    #     output_text=json.load(open(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_questions_s{start}d{end}.json'))    
    
    
    history_list=[]
    for i,text in enumerate(output_text):
        history_list.append(split_dialogue(text))

    print(len(history_list),len(dataset))
    
    dataset=dataset.remove_columns(['history','input_ids'])
    dataset=dataset.add_column(name='history',column=history_list)
    

    dataset.save_to_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage{stage}/s{start}e{end}')

