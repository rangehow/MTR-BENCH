from transformers import AutoTokenizer
import datasets
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main")

dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired')

cnt=0
odd_token_count=0
even_token_count=0
# 遍历数据集的每个元素
for i in tqdm(range(15, len(dataset), 16)):
    cnt+=1
    history = dataset[i]['history']
    
    # 假设 history 是一个列表，可以按索引分成奇偶部分
    even_history = [history[j] for j in range(0, len(history), 2)]  # 奇数位置元素
    odd_history = [history[j] for j in range(1, len(history), 2)]  # 偶数位置元素
    
    # Tokenizing the odd and even histories
    odd_encoded = tokenizer.batch_encode_plus(odd_history,  truncation=True,)
    even_encoded = tokenizer.batch_encode_plus(even_history,  truncation=True)
    
    # Counting the tokens in both even and odd history
    odd_token_count += sum([len(ids) for ids in odd_encoded['input_ids']])
    even_token_count += sum([len(ids) for ids in even_encoded['input_ids']])
    
print(odd_token_count/cnt/8)
print(even_token_count/cnt/8)

