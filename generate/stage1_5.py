import os
import datasets
from transformers import AutoTokenizer
from functools import partial
from torch.cuda import device_count
# import sglang as sgl
import argparse

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.0' # 为了加速torch编译指定的，这个版本仅限于A100，https://en.wikipedia.org/wiki/CUDA
os.environ['SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN']='1'

# Judge_prompt='''Given a text, determine whether it directly states or implies an inability to provide an answer (e.g., "I don't know," "I can't answer that," or similar phrases). If the text explicitly indicates such an inability, return 1. Otherwise, return 0.
# ## text
# {history}
# '''

# def judge_func(instances,prompt,tokenizer):
    
#     token_ids_list=[]
    
#     for query, answer in zip(instances['query'], instances['answer']):
#         if isinstance(query, list):
#             user_query = query[-2]  # 最后一轮的用户输入
#             assistant_answer = query[-1]  # 最后一轮的助手回复
#             history_str = f"{assistant_answer}"

#         else:
#             # 处理单轮对话
#             history_str = f"{answer}"
        
        
#         all_text = prompt.format_map({'history':history_str})
#         token_ids = tokenizer.apply_chat_template([{'role':'user','content':all_text}],add_generation_prompt=True)
#         token_ids_list.append(token_ids)
    

#     return {'input_ids':token_ids_list}






def length(instances, tokenizer):
    lengths = []
    for answer in instances['answer']:  # Use 'answer' key directly for batching
        lengths.append(len(tokenizer.encode(answer)))  # Tokenize and get length
    return {"lengths": lengths}


def parse_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",default=1,type=int)
    return parser.parse_args()

if __name__ == '__main__':
    # Directory paths
    args=parse_args()
    stage=args.stage
    print('stage',stage)
    dataset_folder = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage{stage}'
    new_dataset_folder = f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage{stage}_5'
    
    if not os.path.exists(new_dataset_folder):
        os.makedirs(new_dataset_folder)

    # Variables for overall statistics
    total_origin_length = 0
    total_filtered_length = 0
    total_length_before_filtering = 0
    total_length_after_filtering = 0
    total_datasets = 0

    # Model and tokenizer initialization
    # judge_model = '/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2.5-7B-Instruct/main'
    length_model_path = "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    tokenizer = AutoTokenizer.from_pretrained(length_model_path)
    # judge_tokenizer = AutoTokenizer.from_pretrained(judge_model)

    # sampling_params = {"temperature": 0, 'max_new_tokens': 256}
    # llm = sgl.Engine(model_path=judge_model, dp_size=device_count(), context_length=12800)

    for folder in os.listdir(dataset_folder):
        folder_path = os.path.join(dataset_folder, folder)
        
        if os.path.isdir(folder_path):  # Process only directories
            new_folder_path = os.path.join(new_dataset_folder, folder)
            if os.path.exists(new_folder_path):
                continue
            dataset = datasets.load_from_disk(folder_path)
    
            print(f"\nProcessing dataset from folder: {folder_path}")
            
            # Add length information to the dataset
            dataset = dataset.map(partial(length, tokenizer=tokenizer), batched=True, num_proc=32)
            
            # Length statistics before filtering
            lengths = dataset['lengths']
            avg_length_before = sum(lengths) / len(lengths) if len(lengths) > 0 else 0
            origin_length = len(dataset)
            print(f"Original dataset length: {origin_length}")
            print(f"Average length before filtering: {avg_length_before}")
            
            # Update overall statistics
            total_origin_length += origin_length
            total_length_before_filtering += sum(lengths)
            
            # Filter by length > 120
            dataset = dataset.filter(lambda x: x['lengths'] > 120, num_proc=32)
            
            # Length statistics after filtering
            lengths = dataset['lengths']
            avg_length_after = sum(lengths) / len(lengths) if len(lengths) > 0 else 0
            filtered_length = len(dataset)
            print(f"Filtered dataset length: {filtered_length}")
            print(f"Average length after filtering: {avg_length_after}")
            
            # Update statistics after the second filtering
            total_filtered_length += filtered_length
            total_length_after_filtering += sum(lengths)
            total_datasets += 1


            # # Apply the second filter using the judge model
            # dataset = dataset.map(partial(judge_func, tokenizer=judge_tokenizer, prompt=Judge_prompt), batched=True, num_proc=32)
            # outputs = llm.generate(input_ids=dataset['input_ids'], sampling_params=sampling_params)
            # output_text = [o['text'] for o in outputs]
            
            # output_boolean = [True if '0' in text  else False for text in output_text]
            # trash_dataset =  dataset.filter(lambda x, idx: not output_boolean[idx], with_indices=True, num_proc=32)
            # dataset = dataset.filter(lambda x, idx: output_boolean[idx], with_indices=True, num_proc=32)
    

            
            
            # Save the processed dataset to disk
            
            os.makedirs(new_folder_path, exist_ok=True)
            dataset.save_to_disk(new_folder_path)
            print(f"Processed dataset saved to: {new_folder_path}")
    
    # Final statistics for all datasets
    avg_length_before_all = total_length_before_filtering / total_origin_length if total_origin_length > 0 else 0
    avg_length_after_all = total_length_after_filtering / total_filtered_length if total_filtered_length > 0 else 0

    print("\nSummary of statistics for all datasets processed:")
    print(f"Total datasets processed: {total_datasets}")
    print(f"Total original dataset length: {total_origin_length}")
    print(f"Total filtered dataset length: {total_filtered_length}")
    print(f"Overall average length before filtering: {avg_length_before_all:.2f}")
    print(f"Overall average length after filtering: {avg_length_after_all:.2f}")
