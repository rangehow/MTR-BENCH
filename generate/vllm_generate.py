import asyncio
import logging
import json
import os
from prompts import generate_prompt
from tqdm import tqdm


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)  # 或 logging.ERROR

# 配置从环境变量读取

INPUT_FILE = os.getenv('INPUT_FILE')
OUTPUT_FILE = os.getenv('OUTPUT_FILE')
CACHE_DIR = os.getenv('CACHE_DIR', 'cache')

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 缓存文件路径
CACHE_FILE = os.path.join(CACHE_DIR, "cache_all.jsonl")

def load_cache():
    """加载缓存文件，发现最后一行不完整时直接删除"""
    cache_dict = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r+', encoding='utf-8') as f:
                # 读取所有行
                lines = f.readlines()
                if not lines:
                    return {}
                
                # 检查最后一行是否完整
                last_line = lines[-1].strip()
                if not (last_line.startswith('{') and last_line.endswith('}')):
                    logging.warning(f"Found incomplete last line (removing): {last_line[:100]}...")
                    lines = lines[:-1]  # 删除最后一行
                    # 截断文件并重写
                    f.seek(0)
                    f.truncate()
                    f.writelines(lines)
                
                # 处理所有有效行
                for line in lines:
                    try:
                        cache_item = json.loads(line.strip())
                        cache_dict.update(cache_item)
                    except json.JSONDecodeError as e:
                        logging.error(f"Error decoding cache line: {e}")
                        continue
                
                return cache_dict
        except Exception as e:
            logging.error(f"Error loading cache file: {e}")
            return {}
    return {}

def get_from_cache(text):
    """从缓存中获取结果"""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    cache_dict = load_cache()
    if cache_key in cache_dict:
        logging.info(f"Cache hit: {cache_key}")
        return cache_dict[cache_key]
    return None

def save_to_cache(text, response):
    """保存结果到缓存，以JSONL格式追加"""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    try:
        cache_item = {cache_key: response}
        with open(CACHE_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cache_item, ensure_ascii=False) + '\n')
    except Exception as e:
        logging.error(f"Error saving cache: {e}")

def process_data(file_path):
    """处理输入文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text = []
        for cluster_id, cluster_info in data['clusters'].items():
            documents = cluster_info['documents']
            cluster_text = ""
            for i, doc in enumerate(documents):
                cluster_text += f"[{cluster_id}{i}]{doc}\n"
            text.append(cluster_text)
        return text
    except Exception as e:
        logging.error(f"Error processing input file: {e}")
        raise

def save_to_jsonl(data, filename):
    """保存结果到jsonl文件"""
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for item in data:
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        logging.error(f"Error saving results: {e}")
        raise





        


if __name__ == "__main__":
    text= process_data(INPUT_FILE)
    from datasets import Dataset
    from transformers import AutoTokenizer
    model_name = "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset=Dataset.from_dict({"text":text})
    def chat_template(batch):
        updated_text = [
            tokenizer.apply_chat_template([{"role": "user", "content": generate_prompt(t)[0]}], add_generation_prompt=True)
            for t in batch["text"]
        ]
        return {"text": updated_text}

    dataset=dataset.map(chat_template,batched=True,num_proc=32)
    prompt_token_ids = dataset["text"]


    max_model_len, tp_size = 128000, 4
    
    from vllm import LLM, SamplingParams
    llm = LLM(model=model_name, tensor_parallel_size=tp_size, max_model_len=max_model_len, trust_remote_code=True, enable_prefix_caching=True)

    sampling_params=SamplingParams(repetition_penalty=1.05,top_p=0.8,temperature=0.7,max_tokens=15000)
    outputs = llm.generate(prompt_token_ids=prompt_token_ids, sampling_params=sampling_params)

    generated_text = [output.outputs[0].text for output in outputs]
    save_to_jsonl(generated_text, OUTPUT_FILE)
    