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

PROMPT='''Please propose a single, simple question based on the given documents. The question should be relevant to the content of only one specific document, and it should begin with [Document Number]. Do not mention that the questions are based on documents or external sources. The generated questions should not include expressions like "according to the document.". Avoid combining two or more questions in one. Here is an example:

## Given Documents
[1] In 2004, Trump became the producer and host of NBC's reality TV show "The Apprentice"; in the same year, he produced and starred in the reality show "The Apprentice", where he portrayed himself as a top businessman, and the show was twice nominated for the Best Competition Reality Show. Additionally, he launched his own radio show "Trumped!". In 2007, he was honored with a Hollywood Walk of Fame star in recognition of his television contributions. However, after he assumed the presidency, the star was repeatedly vandalized by the public, and on August 7, 2018, the West Hollywood City Council voted to permanently remove Trump's star. In January 2019, for his appearances in the documentaries "Death of a Nation" and "Fahrenheit 11/9", he was awarded the Worst Actor and Worst Screen Combo at the 39th Golden Raspberry Awards.
[2] On January 20, 2017, Trump was sworn into office in Washington, D.C., officially becoming the 45th President of the United States. After taking office, he began to gradually sign executive orders to fulfill campaign promises, carrying out reforms in employment, taxation, trade, immigration, and foreign policy, achieving promises such as withdrawing from the Trans-Pacific Partnership (TPP), restarting pipeline projects, and repealing the ObamaCare Act. During his presidency, he initiated a trade war with China, withdrew from several international organizations, and pushed for building a border wall with Mexico, leading to the longest government shutdown in US history at the time. On November 6, 2018, the results of the US midterm congressional elections were announced, with the Democrats regaining majority control of the House of Representatives while the Republicans strengthened their majority in the Senate. In September 2019, Trump was accused of pressuring Zelensky during a phone call to investigate his political opponent, leading to an impeachment inquiry initiated by the Democrats (see: Trump impeachment). On September 24, the US House of Representatives launched an impeachment investigation against Trump. On December 18, two articles of impeachment were passed after a full vote. On February 5, 2020, the US Senate voted to acquit Trump on both articles of impeachment.
## Questions
[2] When was Trump impeached?   

Please propose a single, simple question based on the given document. The question should be relevant to the content of only one specific document, and it should begin with [Document Number]. Do not mention that the questions are based on documents or external sources. The generated questions should not include expressions like "according to the document.". Avoid combining two or more questions in one. Here is an example:

## Given Documents
{documents}
## Questions
'''


REASONER_PROMPT="""Based on the provided documents, give a detailed and complete response to the user's question. Answer directly without mentioning any references to documents or sources.
## Given Documents
{document}
## Question
{question}
"""


def add_answer(instances):

    history_list = []
    for history, answer in zip(instances['history'], instances['answer']):
        history_list.append(history+[answer])

    return {'history': history_list}


def first_row_query(instances,prompt,tokenizer):
    
    token_ids_list=[]
    for instance in instances['text']:

        # 假设 instance 是你的文档列表
        document_list = instance[:5]
        # 去除原始文档中的方序号
        cleaned_documents = [re.sub(r'\[\d+\]', '', doc) for doc in document_list]
        # 将文档链接起来并添加序号
        document_str = '\n'.join([f"[{i+1}] {doc}" for i, doc in enumerate(cleaned_documents)])


        all_text = prompt.format_map({'documents':document_str})
        token_ids = tokenizer.apply_chat_template([{'role':'user','content':all_text}],add_generation_prompt=True)
        token_ids_list.append(token_ids)
    

    return {'input_ids':token_ids_list}



def first_row_response(instances,prompt,tokenizer):
    
    token_ids_list=[]
    for history,document_list in zip(instances['history'],instances['document_list']):

        # 去除原始文档中的方序号
        cleaned_documents = [re.sub(r'\[\d+\]', '', doc) for doc in document_list[:5]]
        # 将文档链接起来并添加序号
        document_str = '\n'.join([f"[{i+1}] {doc}" for i, doc in enumerate(cleaned_documents)])

        all_text = prompt.format_map({'document':document_str,'question':history[0]})
        token_ids = tokenizer.apply_chat_template([{'role':'user','content':all_text}],add_generation_prompt=True)
        token_ids_list.append(token_ids)
    

    return {'input_ids':token_ids_list}



def split_questions(text, cid):
    # 正则表达式匹配序号和问题
    pattern = r'\[(\d+)\]\s*(.+?)\s*$'
    
    # 查找所有匹配的内容
    matches = re.findall(pattern, text, re.MULTILINE)

    result = []
    # 只保留一个
    for did, query in matches[:1]:
        result.append({'did': did, 'query': query.strip(), 'cid': cid})
    
    return result


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
    # if os.path.exists(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage1/s{start}e{end}'):
    #     print("数据集已存在，退出")
    #     exit()

    model_path='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-72B-Instruct/main'
    # model_path="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset_list=[]
    for i in range(start,end+1):
        
        dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/cluster_wow/{i}')
        dataset_list.append(dataset)
    
    dataset = datasets.concatenate_datasets(dataset_list)
    dataset = dataset.map(partial(first_row_query,prompt=PROMPT,tokenizer=tokenizer),batched=True,num_proc=32,)
    # sampling_params = {"temperature": 0.8, "top_p": 0.95,'max_new_tokens':10000}
    sampling_params = {"temperature": 0,'max_new_tokens':1024}
    llm = sgl.Engine(model_path=model_path,tp_size=4,dp_size=2 if device_count()>4 else 1,context_length=128000)
    if not os.path.exists(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/output_questions_s{start}d{end}.json'):


        print(tokenizer.decode(dataset[0]['input_ids']))

        outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)
        # import pdb
        # pdb.set_trace()
        print("question done")
        output_text=[o['text'] for o in outputs]
        # 保存到/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp
        output_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/output_questions_s{start}d{end}.json'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_text, f, ensure_ascii=False, indent=4)

    else:
        print("复用了已生成的问题数据集")
        output_text=json.load(open(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/output_questions_s{start}d{end}.json'))    


    if not os.path.exists(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/output_legal_question_s{start}d{end}.json'):
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
                    'history': [js['query']],
                    'cid': js['cid'],
                    'local_did':[js['did']],
                    'gold_did': dataset[js['cid']]['did'][int(js['did'])-1],  # 取出原始的数据集中的did列表，然后还原，但是要注意我们这个local did是为了llm方便从1开始的，所以-1。
                    'gold_document': dataset[js['cid']]['text'][int(js['did'])-1],
                    'document_list': dataset[js['cid']]['text'],
                    'did_list': dataset[js['cid']]['did'],
                }
                legal_json_list.append(new_js)
            except Exception as e:
                print(e)
                print(js)
                print(tokenizer.decode(dataset[js['cid']]['input_ids']))
                ilegal_cnt+=1
        print(f"不合规率{ilegal_cnt/len(json_list)}")
        output_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/output_legal_question_s{start}d{end}.json'
        with open(output_path, 'w') as f:
            json.dump(legal_json_list, f, ensure_ascii=False, indent=4)
    else:
        legal_json_list = json.load(open(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/output_legal_question_s{start}d{end}.json'))    
        print("legal_json_list加载完成")
        
    stage1_question_dataset=datasets.Dataset.from_list(legal_json_list)
    
    

    output_path = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/output_answer_s{start}d{end}.json'

    if not os.path.exists(output_path):

        stage1_question_dataset=stage1_question_dataset.map(partial(first_row_response,prompt=REASONER_PROMPT,tokenizer=tokenizer),batched=True,num_proc=64)
        outputs = llm.generate(input_ids=stage1_question_dataset['input_ids'], sampling_params=sampling_params)
        output_text=[o['text'] for o in outputs]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(output_text, f, ensure_ascii=False, indent=4)
    else:
        output_text=json.load(open(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/tmp/output_answer_s{start}d{end}.json'))
        print("answer加载完成")

    

    print(f'答案生成完成了，保存到了{output_path}')
    stage1_question_dataset = stage1_question_dataset.add_column(name='answer',column=output_text)
    stage1_question_dataset = stage1_question_dataset.map(add_answer,batched=True,num_proc=32,remove_columns=['answer','input_ids'])
    
 
    stage1_question_dataset.save_to_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage1/s{start}e{end}')

