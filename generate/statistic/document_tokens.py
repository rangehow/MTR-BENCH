from transformers import AutoTokenizer
import datasets
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main")

dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean')

cnt=0
char_length = 0
token_lenght = 0
# 遍历数据集的每个元素
for i in tqdm(range(100000)):
    cnt+=1
    char_length+=len(dataset[i]['text'])
    token_lenght+=len(tokenizer(dataset[i]['text'])['input_ids'])
    
print(cnt)
print(char_length)
print(char_length/cnt)
print(token_lenght)
print(token_lenght/cnt)
