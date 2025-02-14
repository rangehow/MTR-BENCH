import datasets
from tqdm import tqdm

dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-train')

new_dataset = []
for i in tqdm(range(3,len(dataset)//2,8)):
    sample= dataset[i]
    latter_data=dataset[len(dataset)//2+i]
    for j in range(i-3,i+1):
        
        new_item= dataset[j]
        new_item['document_list'] = new_item['document_list']+latter_data['document_list']
        new_item['did_list'] = new_item['did_list']+latter_data['did_list']
        new_dataset.append(new_item)
    for j in range(len(dataset)//2+i-3,len(dataset)//2+4+i-3):
        new_item= dataset[j]
        new_item['history'] = sample['history']+new_item['history']
        new_item['local_did'] = sample['local_did']+[str(len(sample['document_list'])+int(d)) for d in new_item['local_did'][:4]]
        new_item['document_list'] = sample['document_list']+new_item['document_list']
        new_item['did_list'] = sample['did_list']+new_item['did_list']

        new_dataset.append(new_item)

  
new_dataset = datasets.Dataset.from_list(new_dataset)

new_dataset.save_to_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-train-paired')


