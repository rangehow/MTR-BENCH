import torch
import datasets
from transformers import Trainer, AutoModel, AutoTokenizer, TrainingArguments
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
# torch.autograd.set_detect_anomaly(True)

from datasets import disable_caching
disable_caching()


def extend_position_embeddings(model, new_max_length):
    """
    扩展 BERT 模型的位置编码矩阵，并更新 position_ids 和 token_type_ids buffer。

    Args:
        model:  预训练的 BERT 模型 (transformers.BertModel)
        new_max_length:  新的最大序列长度
    """
    if new_max_length <= model.config.max_position_embeddings:
        print(f"新的最大长度 {new_max_length} 没有超过当前的 {model.config.max_position_embeddings}，无需扩展。")
        return

    old_position_embeddings = model.embeddings.position_embeddings.weight.data
    old_max_length = model.config.max_position_embeddings
    embedding_dim = old_position_embeddings.size(1)

    # 创建新的位置编码矩阵
    new_position_embeddings = nn.Embedding(new_max_length, embedding_dim).weight.data
    torch.nn.init.normal_(new_position_embeddings, mean=0.0, std=model.config.initializer_range)

    # 复制旧的位置编码
    n = min(old_max_length, new_max_length)
    new_position_embeddings[:n, :] = old_position_embeddings[:n, :]

    # 替换模型中的位置编码权重
    model.embeddings.position_embeddings = nn.Embedding.from_pretrained(new_position_embeddings)

    # **关键修改：重新注册 position_ids 和 token_type_ids buffer**
    model.embeddings.register_buffer(
        "position_ids", torch.arange(new_max_length).expand((1, -1)), persistent=False
    )
    model.embeddings.register_buffer(
        "token_type_ids", torch.zeros((1, new_max_length), dtype=torch.long), persistent=False
    )


    # 更新模型配置
    model.config.max_position_embeddings = new_max_length
    model.embeddings.position_ids.max_len = new_max_length # 确保 position_ids 的 max_len 也更新 (某些模型结构可能需要)


    print(f"位置编码已扩展到 {new_max_length}，position_ids 和 token_type_ids buffer 已更新。")





def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)



def format(item):
    # 合并历史对话，奇偶行区分 User 和 Agent
    anchor = ""
    
    for i, line in enumerate(item["history"][:-1]):
        if i % 2 != 0:
            anchor += "Agent: " + line
        else:
            anchor += "User: " + line

        if i < len(item["history"][:-1]) - 1:
            anchor += "\n"

    positive = item["gold_document"]
    negatives = item["document_list"]
    

    del negatives[int(item["local_did"][-1]) - 1]


    return {
        "anchor": anchor,
        "positive": positive,
        "negative": negatives  # 依然保留一个列表，以便多负例
    }


# 反的
# def format(item):
#     # 合并历史对话，奇偶行区分 User 和 Agent
#     print("反的")
#     anchor = ""
    
#     for i, line in enumerate(item["history"][:-1]):
#         if i % 2 != 0:
#             anchor += "User: " + line
#         else:
#             anchor += "Agent: " + line

#         if i < len(item["history"][:-1]) - 1:
#             anchor += "\n"

#     positive = item["gold_document"]
#     negatives = item["document_list"]
    

#     del negatives[int(item["local_did"][-1]) - 1]


#     return {
#         "anchor": anchor,
#         "positive": positive,
#         "negative": negatives  # 依然保留一个列表，以便多负例
#     }

# 




def load_dataset(path):
    dataset = datasets.load_from_disk(path)
    dataset = dataset.shuffle(seed=42,keep_in_memory=True)
    dataset = dataset.select(range(40000))
    dataset = dataset.filter(lambda x: len(x['document_list'])>4,num_proc=32)
    
    dataset = dataset.map(format, num_proc=32, remove_columns=dataset.column_names)

    
    return dataset

# ---------------------------
# 3) collate_fn: 将批量数据 token 化并返回字典
#    若负例是多个，会展平后再记录每条的负例数量，供 compute_loss 重排用
# ---------------------------
def collate_fn(batch, query_tokenizer,context_tokenizer,max_length):
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives_list = [item["negative"] for item in batch]

    anchor_input = query_tokenizer(
        anchors,
        padding=True,
        truncation=True,     # <-- Add
        max_length=max_length,      # <-- Add
        return_tensors="pt",
    )
    positive_input = context_tokenizer(
        positives,
        padding=True,
        truncation=True,     # <-- Add
        max_length=max_length,      # <-- Add
        return_tensors="pt",
    )

    flattened_negatives = []
    negative_counts = []
    for neg_docs in negatives_list:
        negative_counts.append(len(neg_docs))
        flattened_negatives.extend(neg_docs)

    
    negative_input = context_tokenizer(
        flattened_negatives,
        padding=True,
        truncation=True,     # <-- Add
        max_length=max_length,      # <-- Add
        return_tensors="pt",
    )
    

    return {
        "anchor_input": anchor_input,
        "positive_input": positive_input,
        "negative_input": negative_input,
        "negative_counts": negative_counts,
    }


class BiEncoderTrainer(Trainer):

    def compute_loss(self, model, inputs, return_outputs=False,**kwargs):
        anchor_input = inputs["anchor_input"]
        positive_input = inputs["positive_input"]
        negative_input = inputs["negative_input"]  

        # 编码 anchor 和 positive
        anchor_outputs = model(**anchor_input) 
        anchor_embeds = anchor_outputs.last_hidden_state[:, 0]
        positive_outputs = model(**positive_input) 
        positive_embeds = positive_outputs.last_hidden_state[:, 0]

        
        negative_outputs = model(**negative_input) 
        negative_embeds = negative_outputs.last_hidden_state[:, 0]
        

        split_sizes = inputs["negative_counts"]
        negative_embeds = torch.split(negative_embeds, split_sizes)

        # 计算 InfoNCE 损失 (保持不变)
        losses = []
        for i in range(anchor_embeds.shape[0]):
            loss = info_nce(
                query=anchor_embeds[i].unsqueeze(0),
                positive_key=positive_embeds[i].unsqueeze(0),
                negative_keys=negative_embeds[i],
                temperature=0.1,
            )
            losses.append(loss)
        return torch.mean(torch.stack(losses))


if __name__=='__main__':

    dataset_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-train-paired"
    model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/models/bge-large-en-v1.5"

    max_length=8192
    # 加载模型和 tokenizer
    model = AutoModel.from_pretrained(model_path)
    extend_position_embeddings(model,max_length)
    query_tokenizer= AutoTokenizer.from_pretrained(model_path,truncation_side='left')
    context_tokenizer = AutoTokenizer.from_pretrained(model_path)

    # 加载数据
    dataset = load_dataset(dataset_dir)
    print(dataset[0]['anchor'])

    # 训练参数
    training_args = TrainingArguments(
        output_dir="./model_dir/bge-large-8192-4w-fuck-acc64",
        overwrite_output_dir=True,
        logging_dir="./logs",
        logging_steps=1,
        save_strategy='epoch',
        save_total_limit=2,
        # max_steps=900,
        num_train_epochs=2,
        warmup_ratio=0.05,
        learning_rate=3e-5,
        weight_decay=0.01,
        gradient_accumulation_steps=64,
        per_device_train_batch_size=1,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=5,
        fp16=True,
        lr_scheduler_type="linear",
        remove_unused_columns=False,
    )

    # 初始化 Trainer
    trainer = BiEncoderTrainer(
        model=model,
        args=training_args,
        tokenizer=query_tokenizer,
        train_dataset=dataset,

        data_collator=lambda batch: collate_fn(batch, query_tokenizer,context_tokenizer,max_length)
    )


    trainer.train()
    
