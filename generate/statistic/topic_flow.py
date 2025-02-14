from collections import Counter
import datasets
from tqdm import tqdm

dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-train-paired')

id_mapping = {}  # 用于存储 original_local_did 到 new_id 的映射
next_id = 1     # 用于生成新的 id，起始值为 1
arrays=[]
for i in tqdm(range(7, len(dataset), 8)):
    original_local_did_list = dataset[i]['local_did'] # 获取原始的 local_did 列表
    # print(original_local_did_list)
    re_encoded_local_did_list = [] # 存储重新编码后的 local_did 列表

    for local_did_value in original_local_did_list: # 遍历 local_did 列表中的每个值
        if local_did_value in id_mapping:
            # 如果 local_did_value 已经出现过，则使用已经分配的 id
            new_id = id_mapping[local_did_value]
        else:
            # 如果是新的 local_did_value，则分配一个新的 id
            new_id = next_id # 将 id 转换为字符串，与你的例子输出保持一致
            id_mapping[local_did_value] = new_id # 存储映射关系
            next_id += 1 # id 计数器加 1
        re_encoded_local_did_list.append(new_id) # 将新的 id 添加到重新编码的列表中
    
    # print('before',original_local_did_list)
    # print('after',re_encoded_local_did_list)
    id_mapping = {}
    next_id = 1
    
    arrays.append(re_encoded_local_did_list) # 打印重新编码后的 local_did 列表



def count_flow(arrays):
    """
    统计数组列表中数字的流动次数。

    Args:
      arrays: 一个包含多个数字数组的列表。

    Returns:
      一个字典，键是流动组合 (例如 "1->2")，值是流动次数。
    """
    flow_counts = [Counter() for _ in range(8)]  # 初始化一个字典来存储流动次数

    for array in arrays:
        for i in range(len(array)-1):
            
            flow_counts[i][(int(array[i]),int(array[i+1]))]+=1
    
    return flow_counts


# 调用函数进行统计
flow_results = count_flow(arrays)


# 打印结果
for flow in flow_results:
    print(flow)