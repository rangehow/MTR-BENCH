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

PROMPT='''Please generate a single and simple question based on the provided document and the dialogue history. The question should be relevant to the content of only one specific document and begin with [Document Number]. Do not mention that the questions are based on documents or external sources. The generated questions should not include expressions like 'according to the document.' Encourage the generated questions to depend on different documents from those in the Dialogue History. Do not generate questions that have already appeared in the historical dialogue. Here is an example:

## Given Documents
[1] In 2004, Trump became the producer and host of NBC's reality TV show "The Apprentice"; in the same year, he produced and starred in the reality show "The Apprentice", where he portrayed himself as a top businessman, and the show was twice nominated for the Best Competition Reality Show. Additionally, he launched his own radio show "Trumped!". In 2007, he was honored with a Hollywood Walk of Fame star in recognition of his television contributions. However, after he assumed the presidency, the star was repeatedly vandalized by the public, and on August 7, 2018, the West Hollywood City Council voted to permanently remove Trump's star. In January 2019, for his appearances in the documentaries "Death of a Nation" and "Fahrenheit 11/9", he was awarded the Worst Actor and Worst Screen Combo at the 39th Golden Raspberry Awards.
[2] During the 2016 campaign, Trump promised to repeal and replace the Affordable Care Act. Shortly after taking office as president, he immediately urged Congress to do so. In May 2017, the Republican-controlled House of Representatives passed a party-line vote to repeal the Affordable Care Act and replace it with the Trumpcare Act. However, during the Senate vote, three Republican senators—McCain, Collins, and Murkowski—defected and opposed the repeal, along with all Democratic senators. As a result, the proposal was defeated in the Senate with a vote of 49 to 51.
## Dialogue History
[1] In 2004, Donald Trump became the producer and host of which NBC reality TV show?
Assistant: "The Apprentice"
[1] What significant honor did he receive in 2007 for his contributions to television?
Assistant: In 2007, he was honored with a star on the Hollywood Walk of Fame for his contributions to television.
[2] What did Trump promise to do during the 2016 campaign?
Assistant: Trump promised to repeal and replace the Affordable Care Act.

## Question
[2] Did he sucess fully repeal and replace it? 

Please generate a single and simple question based on the provided document and the dialogue history. The question should be relevant to the content of only one specific document and begin with [Document Number]. Do not mention that the questions are based on documents or external sources. The generated questions should not include expressions like 'according to the document.' Encourage the generated questions to depend on different documents from those in the Dialogue History. Do not generate questions that have already appeared in the historical dialogue.

## Given Documents
{documents}
## Dialogue History
{history}
## Question
'''


REASONER_PROMPT="""Based on the provided document and chat history, give a detailed and complete response to the user's question. Answer directly without mentioning any references to documents or sources. If the question involves multiple aspects, break down the answer into bullet points for clarity.
## Given Document
{document}
## Question
{question}
"""

def history_row_query(instances,prompt,tokenizer):
    
    token_ids_list=[]
    
    for history,document_list,local_did_list in zip(instances['history'],instances['document_list'],instances['local_did']):
        
        # 处理多轮对话历史
        history_str = ""
        for i in range(0, len(history), 2):
            user_query = history[i]
            assistant_answer = history[i+1] if i+1 < len(history) else ""
            history_str += f"[{local_did_list[i//2]}] {user_query}\nAssistant: {assistant_answer}\n"

        # 去除原始文档中的方序号
        cleaned_documents = [re.sub(r'\[\d+\]', '', doc) for doc in document_list[:5]]
        # 将文档链接起来并添加序号
        document_str = '\n'.join([f"[{i+1}] {doc}" for i, doc in enumerate(cleaned_documents)])

        all_text = prompt.format_map({'documents':document_str,'history':history_str})
        token_ids = tokenizer.apply_chat_template([{'role':'user','content':all_text}],add_generation_prompt=True)
        token_ids_list.append(token_ids)
    

    return {'input_ids':token_ids_list}



def history_row_response(instances,prompt,tokenizer):
    
    token_ids_list=[]
    for history,document_list in zip(instances['history'],instances['document_list']):
        messages=[]
        for i in range(0, len(history)-1, 2):
            user_query = history[i]
            assistant_answer = history[i+1]
            messages.append({'role':'user','content':user_query})
            messages.append({'role':'assistant','content':assistant_answer})

        # 去除原始文档中的方序号
        cleaned_documents = [re.sub(r'\[\d+\]', '', doc) for doc in document_list[:5]]
        # 将文档链接起来并添加序号
        document_str = '\n'.join([f"[{i+1}] {doc}" for i, doc in enumerate(cleaned_documents)])

        messages.append( {'role':'user','content':prompt.format_map({'document':document_str,'question':history[-1]})})

        token_ids = tokenizer.apply_chat_template(messages,add_generation_prompt=True)
        token_ids_list.append(token_ids)
    

    return {'input_ids':token_ids_list}


def split_questions(text, cid):
    # 正则表达式匹配序号和问题
    pattern = r'\[(\d+)\]\s*(.+?)\s*$'
    
    # 查找所有匹配的内容
    matches = re.findall(pattern, text, re.MULTILINE)

    result = []
    # 确保每个cid只留一个
    for did, query in matches[:1]:
        result.append({'did': did, 'query': query.strip(), 'cid': cid})
    
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
    sampling_params = {"temperature": 0,'presence_penalty':0.1,'max_new_tokens':10000}
    llm = sgl.Engine(model_path=model_path,tp_size=4,dp_size=2 if device_count()>4 else 1,context_length=128000)
    if not os.path.exists(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_questions_s{start}d{end}.json'):


        print(tokenizer.decode(dataset[0]['input_ids']))
        outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)

        print("question done")
        output_text=[o['text'] for o in outputs]
        # 保存到/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp
        output_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_questions_s{start}d{end}.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_text, f, ensure_ascii=False, indent=4)

    else:
        print("复用了已生成的问题数据集")
        output_text=json.load(open(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_questions_s{start}d{end}.json'))    
    
    if not os.path.exists(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_legal_question_s{start}d{end}.json'):
        json_list=[]
        for cid,text in enumerate(output_text):
            json_list.extend(split_questions(text,cid))

        legal_json_list = []
        ilegal_cnt=0
        
        # query, ground_truth document, g did, document list, did list, cid 
        for i, js in tqdm(enumerate(json_list)):
            # json_list is list[list[json]] (since LLM can parallel generate multiple query at a time)
            try:
                # dataset的行号和cid是一致的
                new_js = {
                    'cid': js['cid'],
                    'local_did':dataset[js['cid']]['local_did'] +[js['did']],
                    'gold_did': dataset[js['cid']]['did_list'][int(js['did'])-1],  
                    'history': dataset[js['cid']]['history']+[js['query']],
                    'gold_document': dataset[js['cid']]['document_list'][int(js['did'])-1],
                    'document_list': dataset[js['cid']]['document_list'],
                    'did_list': dataset[js['cid']]['did_list']
                }
                legal_json_list.append(new_js)
            except Exception as e:
                print(e)
                import pdb
                pdb.set_trace()
                print(js)
                print(tokenizer.decode(dataset[js['cid']]['input_ids']))
                ilegal_cnt+=1
        print(f"不合规率{ilegal_cnt/len(json_list)}")
        output_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_legal_question_s{start}d{end}.json'
        with open(output_path, 'w') as f:
            json.dump(legal_json_list, f, ensure_ascii=False, indent=4)
    else:
        legal_json_list = json.load(open(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_legal_question_s{start}d{end}.json'))    
        print("legal_json_list加载完成")
        
    stage2_question_dataset=datasets.Dataset.from_list(legal_json_list)
    
    

    output_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_answer_s{start}d{end}.json'

    if not os.path.exists(output_path):
        
        stage2_question_dataset=stage2_question_dataset.map(partial(history_row_response,prompt=REASONER_PROMPT,tokenizer=tokenizer),batched=True,num_proc=1)
        outputs = llm.generate(input_ids=stage2_question_dataset['input_ids'], sampling_params=sampling_params)
        output_text=[o['text'] for o in outputs]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_text, f, ensure_ascii=False, indent=4)
    else:
        output_text=json.load(open(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/stage{stage}_output_answer_s{start}d{end}.json'))
        print("answer加载完成")

    
    
    print(f'答案生成完成了，保存到了{output_path}')
    
    stage2_question_dataset = stage2_question_dataset.add_column(name='answer',column=output_text)
    stage2_question_dataset = stage2_question_dataset.map(add_answer,batched=True,num_proc=32,remove_columns=['answer','input_ids'])
    # stage2_question_dataset = stage2_question_dataset.remove_columns('answer')

    stage2_question_dataset.save_to_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage{stage}/s{start}e{end}')

