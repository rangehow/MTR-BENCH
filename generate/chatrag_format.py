import datasets






def format(instance):
    messages = []
    
    # dataset_a[instance['did_list'][int(instance['local_did'][-1])-1]]['id']
    ground_truth_ctx = {'ctx':instance['gold_document'],'index':int(instance['gold_did'])}
    for i, line in enumerate(instance["messages"][:-1]):
        if i % 2 != 0:
            messages.append({"role":"assistant","content":line})
            
        else:
            
            messages.append({"role":"user","content":line})

    
    return {'messages':messages,'ground_truth_ctx':ground_truth_ctx,'answer':instance["messages"][-1]}


dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test')






dataset = dataset.rename_column('history','messages')
dataset = dataset.map(format,num_proc=1,remove_columns=dataset.column_names)
# dataset = dataset.remove_columns(['cid','document_list','local_did','did_list'])




dataset.save_to_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-chatrag-format')