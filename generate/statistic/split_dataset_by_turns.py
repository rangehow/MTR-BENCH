import datasets
import os

dataset_folder='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-hard-turn'

origin_dataset=datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired-chatrag-format-hard')


new_dataset=[[] for _ in range(8)]

for i in range(len(origin_dataset)):
    new_dataset[i%8].append(origin_dataset[i])

for i,dataset in enumerate(new_dataset):
    datasets.Dataset.from_list(dataset).save_to_disk(os.path.join(dataset_folder,f'turn_{i}'))
