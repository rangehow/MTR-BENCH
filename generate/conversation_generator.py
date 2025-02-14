import asyncio
import logging
import json
import os
from functools import wraps
import httpx
import hashlib
from openai import AsyncOpenAI
from prompts import generate_prompt
from tqdm import tqdm

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
        logging.FileHandler("debug.log", encoding='utf-8')  # 输出到文件
    ]
)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)  # 或 logging.ERROR

# 配置从环境变量读取
API_KEY = os.getenv('OPENAI_API_KEY')
BASE_URL = os.getenv('OPENAI_BASE_URL') 
INPUT_FILE = os.getenv('INPUT_FILE')
OUTPUT_FILE = os.getenv('OUTPUT_FILE')
CACHE_DIR = os.getenv('CACHE_DIR', 'cache')

# 确保缓存目录存在
os.makedirs(CACHE_DIR, exist_ok=True)

# 缓存文件路径
CACHE_FILE = os.path.join(CACHE_DIR, "cache_all.jsonl")

def load_cache():
    """加载缓存文件，发现最后一行不完整时直接删除"""
    logging.debug("开始加载缓存文件")
    cache_dict = {}
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r+', encoding='utf-8') as f:
                lines = f.readlines()
                logging.debug(f"缓存文件共有 {len(lines)} 行")
                if not lines:
                    logging.info("缓存文件为空")
                    return {}
                
                # 检查最后一行是否完整
                last_line = lines[-1].strip()
                if not (last_line.startswith('{') and last_line.endswith('}')):
                    logging.warning(f"发现不完整的最后一行，正在移除: {last_line[:100]}...")
                    lines = lines[:-1]  # 删除最后一行
                    f.seek(0)
                    f.truncate()
                    f.writelines(lines)
                    logging.info("不完整的最后一行已移除")
                
                # 处理所有有效行
                for idx, line in enumerate(lines):
                    try:
                        cache_item = json.loads(line.strip())
                        cache_dict.update(cache_item)
                        if idx % 1000 == 0:
                            logging.debug(f"已加载 {idx} 行缓存")
                    except json.JSONDecodeError as e:
                        logging.error(f"解码缓存行时出错: {e} - 行内容: {line[:100]}...")
                        continue
                
                logging.info(f"缓存加载完成，总缓存项数: {len(cache_dict)}")
                return cache_dict
        except Exception as e:
            logging.error(f"加载缓存文件时出错: {e}")
            return {}
    logging.info("缓存文件不存在，返回空字典")
    return {}

def get_from_cache(text):
    """从缓存中获取结果"""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    cache_dict = load_cache()
    if cache_key in cache_dict:
        logging.info(f"缓存命中: {cache_key}")
        return cache_dict[cache_key]
    logging.debug(f"缓存未命中: {cache_key}")
    return None

def save_to_cache(text, response):
    """保存结果到缓存，以JSONL格式追加"""
    cache_key = hashlib.md5(text.encode()).hexdigest()
    try:
        cache_item = {cache_key: response}
        with open(CACHE_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(cache_item, ensure_ascii=False) + '\n')
        logging.debug(f"已保存到缓存: {cache_key}")
    except Exception as e:
        logging.error(f"保存缓存时出错: {e}")

# 限制并发请求的装饰器
def limit_async_func_call(max_size: int):
    sem = asyncio.Semaphore(max_size)
    def final_decro(func):
        @wraps(func)
        async def wait_func(*args, **kwargs):
            async with sem:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    logging.error(f"{func.__name__} 异常: {e}")
                    return str(e)
        return wait_func
    return final_decro

# 设置HTTP客户端
custom_http_client = httpx.AsyncClient(
    limits=httpx.Limits(max_connections=4096, max_keepalive_connections=1024),
    timeout=httpx.Timeout(timeout=None)
)

# 设置OpenAI客户端 
openai_async_client = AsyncOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    http_client=custom_http_client
)

def process_data(file_path):
    """处理输入文件"""
    logging.debug(f"开始处理输入文件: {file_path}")
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
        logging.info(f"输入文件处理完成，共生成 {len(text)} 条文本")
        return text
    except Exception as e:
        logging.error(f"处理输入文件时出错: {e}")
        raise

def save_to_jsonl(data, filename):
    """保存结果到jsonl文件"""
    logging.debug(f"开始保存结果到文件: {filename}")
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            for idx, item in enumerate(data):
                json.dump(item, f, ensure_ascii=False)
                f.write('\n')
                if idx % 1000 == 0:
                    logging.debug(f"已保存 {idx} 条结果")
        logging.info(f"结果成功保存到 {filename}")
    except Exception as e:
        logging.error(f"保存结果时出错: {e}")
        raise

# 启用并发限制装饰器（根据需要调整 max_size）
@limit_async_func_call(max_size=500)
async def custom_model_if_cache(text, system_prompt=None, history_messages=[], **kwargs):
    """发送请求到OpenAI API，带缓存机制"""
    logging.debug(f"处理文本: {text[:50]}...")  # 仅记录前50个字符
    try:
        prompt, selected_keys = generate_prompt(text=text)
        logging.debug(f"生成提示词: {prompt[:50]}...")  # 仅记录前50个字符

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.extend(history_messages)
        messages.append({"role": "user", "content": prompt})

        logging.debug("发送请求到 OpenAI API")
        response = await openai_async_client.chat.completions.create(
            model="gpt-4o-eva",
            messages=messages,
            temperature=0.9,        # 高创造性
            frequency_penalty=0.5,  # 增加用词多样性
            **kwargs
        )
        result = response.choices[0].message.content
        logging.debug("收到 OpenAI API 响应")

        # 保存到缓存
        save_to_cache(text, result)
        return result
    except Exception as e:
        # logging.error(f"请求 OpenAI API 时出错: {e}")
        return str(e)

async def send_request():
    """发送并发请求"""
    logging.info("开始发送请求")
    text = process_data(INPUT_FILE)
    TOTAL_REQUESTS = len(text)
    retry_count = 1

    logging.info(f"总请求数: {TOTAL_REQUESTS}")
    with tqdm(total=TOTAL_REQUESTS, desc="Total Progress", position=0, leave=True) as total_pbar:
        while True:
            logging.debug("加载当前缓存")
            cache_dict = load_cache()
            cache_size = len(cache_dict)
            logging.info(f"当前缓存大小: {cache_size}")
            retry_count += 1

            # 更新总进度条到当前缓存大小
            total_pbar.n = min(cache_size, TOTAL_REQUESTS)
            total_pbar.refresh()

            if cache_size >= TOTAL_REQUESTS:
                logging.info(f"缓存已达到目标大小: {cache_size}")
                # 返回前TOTAL_REQUESTS个缓存结果
                results = []
                for idx, text_item in enumerate(text[:TOTAL_REQUESTS]):
                    cache_key = hashlib.md5(text_item.encode()).hexdigest()
                    if cache_key in cache_dict:
                        results.append(cache_dict[cache_key])
                    else:
                        logging.warning(f"未找到缓存项: {cache_key} (文本索引: {idx})")
                return results
            
            # 找出未缓存的文本
            uncached_texts = [text_item for text_item in text[:TOTAL_REQUESTS] 
                              if hashlib.md5(text_item.encode()).hexdigest() not in cache_dict]
            logging.info(f"未缓存的文本数量: {len(uncached_texts)}")

            if not uncached_texts:
                logging.warning("没有未缓存的文本，但缓存大小仍小于总请求数。可能存在问题。")
                break  # 防止无限循环

            # 处理未缓存的文本
            tasks = [custom_model_if_cache(text=text_item) for text_item in uncached_texts]
            logging.info(f"创建了 {len(tasks)} 个任务")

            # 使用异步 tqdm 监控任务完成情况
            with tqdm(total=len(tasks), desc=f"尝试 #{retry_count-1}", position=1, leave=True) as try_pbar:
                for coro in asyncio.as_completed(tasks):
                    try:
                        result = await coro
                        if result:
                            try_pbar.update(1)
                            total_pbar.update(1)
                    except Exception as e:
                        # logging.error(f"任务执行时出错: {e}")
                        try_pbar.update(1)
                        total_pbar.update(1)
            
            logging.info(f"\n\n请求回合 #{retry_count}")

    logging.info("发送请求完成")
    return []

async def cleanup():
    """清理资源"""
    logging.debug("开始清理 HTTP 客户端资源")
    await custom_http_client.aclose()
    logging.info("HTTP 客户端资源已清理")

def main():
    """主函数"""
    logging.info("程序启动")
    try:
        # 验证必需的环境变量
        if not all([API_KEY, BASE_URL, INPUT_FILE, OUTPUT_FILE]):
            raise ValueError("缺少必要的环境变量 (OPENAI_API_KEY, OPENAI_BASE_URL, INPUT_FILE, OUTPUT_FILE)")

        logging.info("环境变量验证通过")
        
        # 运行主要任务
        results = asyncio.run(send_request())
        
        # 保存结果
        save_to_jsonl(results, OUTPUT_FILE)
        logging.info(f"结果已保存到 {OUTPUT_FILE}")
        
        # 清理资源
        asyncio.run(cleanup())
        logging.info("资源清理完成")
        
    except Exception as e:
        logging.error(f"主函数错误: {e}", exc_info=True)
        raise
    finally:
        logging.info("程序结束")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.info("用户中断程序")
    except Exception as e:
        logging.error(f"程序异常终止: {e}", exc_info=True)
        exit(1)
