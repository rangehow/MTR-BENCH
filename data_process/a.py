from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling,Transformer
from transformers import AutoModel, AutoTokenizer
# 载入模型
model_path = '/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/cerebras/Dragon-DocChat-Context-Encoder/main'
model = SentenceTransformer(model_path)

model1 = Transformer(model_path)
pooling_model = Pooling(model1.get_word_embedding_dimension(),pooling_mode='cls')
model2 = SentenceTransformer(modules=[model1, pooling_model])
import pdb
pdb.set_trace()
# 输入文本
texts = ["This is a test text!"]

# 生成embedding
embedding_st = model.encode(texts)

print("SentenceTransformer embedding:", embedding_st)


import torch

# 载入模型和tokenizer
model_path = '/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/cerebras/Dragon-DocChat-Context-Encoder/main'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModel.from_pretrained(model_path)

# 输入文本
texts = ["This is a test text!"]

# Tokenize 输入文本
inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

# 获取模型输出（通常是hidden states）
with torch.no_grad():
    outputs = model(**inputs)

# 获取最后一层hidden state作为文本的embedding（通常是[CLS]标记对应的向量）
embedding_hf = outputs.last_hidden_state.mean(dim=1)  # 可以根据需要选择[CLS]向量，或者mean

print("HuggingFace AutoModel embedding:", embedding_hf)
from sklearn.metrics.pairwise import cosine_similarity

# 计算余弦相似度
similarity = cosine_similarity(embedding_st, embedding_hf.numpy())

print("Cosine Similarity:", similarity)
