import datasets


dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-train-paired')

total = 0
cnt=0
for i in range(15,len(dataset),16):
    cnt+=1
    total+=len(set(dataset[i]['local_did']))

print(total)
print(cnt)
print(total/cnt)