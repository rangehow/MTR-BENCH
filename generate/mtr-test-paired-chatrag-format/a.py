import datasets




dataset_a=datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired-chatrag-format')

dataset_b=datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired-chatrag-format-hard')

for i in range(len(dataset_a)):
    print(dataset_a[i]['messages'][-1])
    print(dataset_b[i]['messages'][-1])
    import pdb
    pdb.set_trace()
# print(dataset_a[4]['messages'])
# print()
# print(dataset[2]['messages'])