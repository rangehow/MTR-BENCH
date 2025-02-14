
import os
import datasets
from tqdm import tqdm
import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


stage_base_dir = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/'

output_dir= '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test'


# 获取所有文件和文件夹
all_files = os.listdir(stage_base_dir)

# 筛选出以 'stage' 开头的文件夹
stage_folders = [os.path.join(stage_base_dir,f) for f in all_files if f.startswith('stage') and os.path.isdir(os.path.join(stage_base_dir, f))]
sorted_stages = sorted(stage_folders)
print(sorted_stages)

# all_datasets = sorted(os.listdir(sorted_stages[-1]),key=natural_sort_key)
all_datasets = [f's{i}e{i}' for i in range(46,51)]
print(all_datasets)

# 遍历每个 stage 文件夹
new_dataset = []
for dataset_name in tqdm(all_datasets):

    temp_datasets = []
    
    for stage in sorted_stages:
        temp_datasets.append(datasets.load_from_disk(os.path.join(stage, dataset_name)))

    for i in range(len(temp_datasets[0])):    
        for j in range(len(temp_datasets)):
            new_dataset.append(temp_datasets[j][i])

    

    
new_dataset = datasets.Dataset.from_list(new_dataset)
new_dataset.save_to_disk(output_dir)