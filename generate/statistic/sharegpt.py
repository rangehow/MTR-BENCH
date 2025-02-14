import json
from transformers import AutoTokenizer
from tqdm import tqdm
data= json.load(open("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/RyokoAI/ShareGPT52K/main/sg_90k_part1.json"))

tokenizer = AutoTokenizer.from_pretrained("/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main")

token_length=0
cnt=0
turn_length=0
for d in tqdm(data):
    cnt+=1
    turn_length+=len(d['conversations'])
    for c in d['conversations']:
        if c['from']=='gpt':
            cnt+=1
            token_length+=len(tokenizer.encode(c['value']))

print(turn_length/cnt/2)
print(token_length)
print(cnt)
print(token_length/cnt)