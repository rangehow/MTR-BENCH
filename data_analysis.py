import datasets
import pdb
from typing import List
from transformers import AutoTokenizer

chatrag_bench = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main'
llama_dir='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main'
tokenizer=AutoTokenizer.from_pretrained(llama_dir)
tasks = ['doc2dial', 'topiocqa', 'inscit', 'quac', 'qrecc']

for task in tasks:
    try:
        dataset = datasets.load_dataset(chatrag_bench, task)['dev']
    except KeyError:
        dataset = datasets.load_dataset(chatrag_bench, task)['test']
    
    total_length = 0  # To accumulate the total length of the shortest answers
    total_count = 0  # To count the number of instances
    
    for item in dataset:
        answers: List[str] = item['answers']
        # Find the shortest answer
        length_list=list(map(lambda x:len(tokenizer.encode(x,add_special_tokens=False)),answers))
        
        # Calculate the length of the shortest answer
        total_length += min(length_list)
        total_count += 1
    
    # Calculate and print the average length
    average_length = total_length / total_count if total_count > 0 else 0
    print(f"Task: {task}, Average Length of Shortest Answer: {average_length}")
