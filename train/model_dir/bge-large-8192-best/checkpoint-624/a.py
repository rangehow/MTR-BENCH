import torch
from transformers import AutoModel

# 模型路径
model_path1 = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/models/bge-large-en-v1.5"
model_path2 = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-wrong/checkpoint-624"

try:
    # 加载模型 1
    model1 = AutoModel.from_pretrained(model_path1)
    position_embeddings1 = model1.embeddings.position_embeddings.weight
    
    # 加载模型 2
    model2 = AutoModel.from_pretrained(model_path2)
    position_embeddings2 = model2.embeddings.position_embeddings.weight
    print(position_embeddings1,position_embeddings2)
    # 提取前 512 行
    position_embeddings1_512 = position_embeddings1[:1024, :]
    position_embeddings2_512 = position_embeddings2[:1024, :]

    # 比较矩阵
    if torch.equal(position_embeddings1_512, position_embeddings2_512):
        print("两个模型的位置嵌入矩阵的前512个位置是相同的。")
    else:
        print("两个模型的位置嵌入矩阵的前512个位置是不同的。")

except Exception as e:
    print(f"在比较过程中出现错误: {e}")
    print("请检查模型路径是否正确，以及模型是否是基于 transformers 的结构。")