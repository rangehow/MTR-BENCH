import datasets


dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-train')
dataset = dataset.filter(lambda x: len(x['document_list'])<2,num_proc=32)
print(len(dataset))

exit()
dataset_a=datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean')


dataset = dataset.remove_columns(['cid','document_list','local_did','did_list'])
dataset = dataset.rename_column('history','messages')
dataset = dataset.map(format,num_proc=32,remove_columns=dataset.column_names)

dataset.save_to_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired-chatrag-format')