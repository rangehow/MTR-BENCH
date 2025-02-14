import sglang as sgl
from transformers import AutoTokenizer
import json
import os
from torch.cuda import device_count


os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0' # 为了加速torch编译指定的，这个版本仅限于A100，https://en.wikipedia.org/wiki/CUDA
os.environ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN']='1'

PROMPT='''Rewrite the following search query to be more effective for information retrieval. Focus on clarity, specificity, and using keywords that will likely return the most relevant results. Output rewrite query only.
'''


def history_row_query(instances, prompt, tokenizer):
    
    token_ids_list = []
    
    for instance in instances:
        
        prompt_str = prompt
        # 处理多轮对话历史
        history_str = ""
        for message in instance['messages'][:-1]:
            
            if message['role'] == 'user':
                history_str += f"User: {message['content']}\n"
            else:
                history_str += f"Assistant: {message['content']}\n"
            

        if history_str != "":
            prompt_str += f"\nDialogue History:\n{history_str}"
        prompt_str += f"\nOriginal Query:\n{instance['messages'][-1]['content']}\n"
        prompt_str += f"Rewrite Query:"
        
        token_ids = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt_str}], add_generation_prompt=True)
        token_ids_list.append(token_ids)

    return token_ids_list


if __name__ == "__main__":

    doc = [
        '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/dataset/ChatRAG-Bench/main/data/topiocqa/modified_dev.json',
        # '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/dataset/ChatRAG-Bench/main/data/qrecc/test.json',
        # '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/dataset/ChatRAG-Bench/main/data/quac/test.json',
        # '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/dataset/ChatRAG-Bench/main/data/doc2dial/test.json'
    ]

    model_path = '/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-72B-Instruct/main'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = {"temperature": 0,  'max_new_tokens': 512}
    llm = sgl.Engine(model_path=model_path, tp_size=4, dp_size=2 if device_count() > 4 else 1, context_length=12800)
    
    for d in doc:
        data = json.load(open(d))
        
        print("数据集列名", data[0].keys())
        print("数据集数量", len(data))

        input_ids = history_row_query(data, PROMPT, tokenizer=tokenizer)

        # print(tokenizer.decode(data[0]['input_ids']))
        outputs = llm.generate(input_ids=input_ids, sampling_params=sampling_params)

        print("question done")
        
        # Generate output text
        output_text = [o['text'] for o in outputs]
        
        # Add the output text to the original data
        for i, instance in enumerate(data):
            data[i]['rewrite_query']=output_text[i]
            

        # Save the updated data to a new file
        new_file_path = d.replace('.json', '_rewrite.json')
        with open(new_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        print(f"Updated data saved to {new_file_path}")
