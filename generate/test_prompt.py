import pdb
import datasets

dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage5/s2e2')

Rewrite_Prompt=""

def formate(instances,prompt):
    for history in instances['history']:
        formate_string=""
        for i in range(len(history)//2):
            formate_string+=f"User: {history[i]}\n"
            formate_string+=f"Assistant: {history[i+1]}\n"
        print(formate_string)
        pdb.set_trace()

dataset = dataset.map(formate,batched=True)


pdb.set_trace()