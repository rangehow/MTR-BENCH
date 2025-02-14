import datasets
import os

# 注意路径要加引号，并建议定义为变量方便复用
folder_path = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage6'
all_path=[ '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage6_old', '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage6_llama_refix','/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage6_llama']

# 获取目录列表并过滤非文件夹项（假设数据集以文件夹形式存储）
subdirs = [d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]

def load_dataset_list(all_path,subdir):
    dataset_list=[]
    for path in all_path:
        full_path =os.path.join(path, subdir)
        dataset_list.append(datasets.load_from_disk(full_path))
    return dataset_list

for subdir in subdirs:
    # 拼接完整路径
    full_path = os.path.join(folder_path, subdir)
   
    # 加载数据集
    dataset = datasets.load_from_disk(full_path)
    dataset_list=load_dataset_list(all_path,subdir)
    for i in range(len(dataset)):
        if len(dataset[i]['history'])!=10:
            break_flag=False
            for j in range(len(dataset_list)):
                    if break_flag:
                        break
                    if len(dataset_list[j][i]['history'])==10:
                        dataset[i]['history']=dataset_list[j][i]['history']
                        break_flag=True
            if break_flag==False:
                print(subdir,)
                for j in range(len(dataset_list)):
                    print(len(dataset_list[j][i]['history']))
                    print(dataset_list[j][i]['history'])
                    print('---')
                import pdb
                pdb.set_trace()

                
                
    # filter_dataset = dataset.filter(lambda x:len(x['history'])==10,num_proc=32)
   

            
        
        
    