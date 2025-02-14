import os
import accelerate
import torch
import datasets
from transformers import AutoModel, AutoTokenizer, TrainingArguments
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from accelerate import Accelerator # 导入 Accelerator
from torch.optim import AdamW # 显式导入 AdamW 优化器
from accelerate import DistributedDataParallelKwargs

from datasets import disable_caching
disable_caching()
# torch.autograd.set_detect_anomaly(True)


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
            raise ValueError(f"<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'. {negative_keys.dim()}")
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
            anchor += "agent: " + line
        else:
            anchor += "user: " + line

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


def load_dataset(path):
    dataset = datasets.load_from_disk(path)
    dataset = dataset.shuffle(seed=42,keep_in_memory=True)
    dataset = dataset.select(range(40000))
    dataset = dataset.filter(lambda x: len(x['document_list'])>4,num_proc=32,load_from_cache_file=False)
    dataset = dataset.map(format, num_proc=32, remove_columns=dataset.column_names,load_from_cache_file=False)
    print(len(dataset))
    return dataset

def collate_fn(batch, query_tokenizer, context_tokenizer):
    anchors = [item["anchor"] for item in batch]
    positives = [item["positive"] for item in batch]
    negatives_list = [item["negative"] for item in batch]

    anchor_input = query_tokenizer(
        anchors,
        padding=True,
        return_tensors="pt",
    )
    positive_input = context_tokenizer(
        positives,
        padding=True,
        return_tensors="pt",
    )

    flattened_negatives = []
    negative_counts = []
    for neg_docs in negatives_list:
        negative_counts.append(len(neg_docs))
        flattened_negatives.extend(neg_docs)

    if len(flattened_negatives) > 0:
        negative_input = context_tokenizer(
            flattened_negatives,
            padding=True,
            return_tensors="pt",
        )
    else:
        negative_input = None

    return {
        "anchor_input": anchor_input,
        "positive_input": positive_input,
        "negative_input": negative_input,
        "negative_counts": negative_counts,
    }



if __name__=='__main__':

    dataset_dir = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-train-paired"
    query_encoder_path = "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-query-encoder/main"
    context_encoder_path = "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-context-encoder/main"

    # 加载数据
    dataset = load_dataset(dataset_dir)
    print(dataset[0]['anchor'])

    dataset.save_to_disk('for_abu')
    


    # 加载模型和 tokenizer

    query_tokenizer= AutoTokenizer.from_pretrained(query_encoder_path, truncation_side='left')
    context_tokenizer = AutoTokenizer.from_pretrained(context_encoder_path)

    
    


    query_model = AutoModel.from_pretrained(query_encoder_path)
    context_model = AutoModel.from_pretrained(context_encoder_path)

    extend_position_embeddings(query_model, 8192)
    extend_position_embeddings(context_model, 8192)

    

    

    # 训练参数 (部分参数可以直接使用，例如 batch_size, learning_rate 等)
    training_args = TrainingArguments(
        output_dir="./model_dir/dragon",
        overwrite_output_dir=True,
        logging_steps=1,
        save_strategy='no', #  修改为不按 epoch 保存，只在训练结束时保存最终模型
        save_total_limit=2,
        warmup_ratio=0.01,
        learning_rate=3.0e-5,
        weight_decay=0.01,
        adam_epsilon=1e-8,
        num_train_epochs=2,
        gradient_accumulation_steps=8,
        per_device_train_batch_size=2,
        dataloader_num_workers=8,
        dataloader_prefetch_factor=5,
        fp16=True,
        lr_scheduler_type="linear",
        logging_first_step=True,
        remove_unused_columns=False,
    )

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # 初始化 Accelerator
    accelerator = Accelerator(mixed_precision="fp16" if training_args.fp16 else "no",gradient_accumulation_steps=training_args.gradient_accumulation_steps,kwargs_handlers=[ddp_kwargs]) # 根据 TrainingArguments 选择 fp16 后端

    # 准备优化器
    optimizer_query = AdamW(
        query_model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        eps=training_args.adam_epsilon,

    )
    optimizer_context = AdamW(
        context_model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay,
        eps=training_args.adam_epsilon,

    )


    # 准备数据加载器
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=True,
        batch_size=training_args.per_device_train_batch_size,
        collate_fn=lambda batch: collate_fn(batch, query_tokenizer, context_tokenizer),
        num_workers=training_args.dataloader_num_workers,
        prefetch_factor=training_args.dataloader_prefetch_factor
    )


    # 使用 accelerator.prepare 对模型、优化器和数据加载器进行处理
    query_model, context_model, optimizer_query, optimizer_context, train_dataloader = accelerator.prepare(
        query_model, context_model, optimizer_query, optimizer_context, train_dataloader
    )

    # 学习率调度器 (如果需要，可以添加学习率预热)
    lr_scheduler_query = torch.optim.lr_scheduler.LinearLR(optimizer_query, start_factor=training_args.warmup_ratio, total_iters=int(len(train_dataloader) * training_args.num_train_epochs * training_args.warmup_ratio)) # 线性预热
    lr_scheduler_context = torch.optim.lr_scheduler.LinearLR(optimizer_context, start_factor=training_args.warmup_ratio, total_iters=int(len(train_dataloader) * training_args.num_train_epochs * training_args.warmup_ratio)) # 线性预热
    


    # 训练循环
    num_update_steps_per_epoch = len(train_dataloader) // training_args.gradient_accumulation_steps
    num_training_steps = int(training_args.num_train_epochs * num_update_steps_per_epoch)

    total_batch_size_fp16 = training_args.per_device_train_batch_size * accelerator.num_processes * training_args.gradient_accumulation_steps

    print("-----------  Training Arguments -----------")
    print(f"  Num steps each epoch = {num_update_steps_per_epoch}")
    print(f"  Num training steps = {num_training_steps}")
    print(f"  Total train batch size (FP16) = {total_batch_size_fp16}")
    print(f"  Gradient Accumulation steps = {training_args.gradient_accumulation_steps}")
    print("------------------------------------------")


    starting_epoch = 0

    progress_bar = tqdm(range(num_training_steps), disable=not accelerator.is_main_process)
    progress_bar.set_description("Training")

    query_model.train()
    context_model.train()
    for epoch in range(starting_epoch, training_args.num_train_epochs):
        
        avg_log_loss = []
        for step, batch in enumerate(train_dataloader):
            # 前向传播
            with accelerator.accumulate([query_model,context_model]):

                anchor_input = batch["anchor_input"]
                positive_input = batch["positive_input"]
                negative_input = batch["negative_input"]
                negative_counts = batch["negative_counts"]


                query_outputs = query_model(**anchor_input)
                positive_outputs = context_model(**positive_input)

                query_embeddings = query_outputs.last_hidden_state[:, 0]
                positive_embeddings = positive_outputs.last_hidden_state[:, 0]


                negative_outputs = context_model(**negative_input)
                negative_embeddings = negative_outputs.last_hidden_state[:, 0]
                negative_embeddings_reshaped = torch.split_with_sizes(negative_embeddings, negative_counts)
                negative_embeddings_list = list(negative_embeddings_reshaped)

                loss = 0.0
                for i in range(len(query_embeddings)):
                    current_query_embedding = query_embeddings[i].unsqueeze(0)
                    current_positive_embedding = positive_embeddings[i].unsqueeze(0)
                    current_negative_embeddings = negative_embeddings_list[i]

                    current_loss = info_nce(
                        current_query_embedding,
                        current_positive_embedding,
                        current_negative_embeddings,
                        temperature=0.1,
                        negative_mode='unpaired'
                    )
                    loss += current_loss
                loss = loss / len(query_embeddings)
                avg_log_loss.append(loss.item())
                

                
                accelerator.backward(loss)
                optimizer_query.step()
                optimizer_context.step()
                lr_scheduler_query.step() # 学习率调度器
                lr_scheduler_context.step() # 学习率调度器
                optimizer_query.zero_grad()
                optimizer_context.zero_grad()

                if accelerator.sync_gradients:
                    avg_log_loss_tensor = torch.tensor(avg_log_loss, device=accelerator.device)  # 转换为Tensor
                    print({"loss_train": torch.mean(accelerator.gather(avg_log_loss_tensor)) / accelerator.gradient_accumulation_steps})
                    avg_log_loss = []  # 重置为列表

                progress_bar.update(1)

        starting_epoch+=1


        # epoch 结束时保存模型 (可以根据需要修改保存策略)
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1} completed")
            # 修改 epoch 保存逻辑，分别保存到子文件夹
            
            query_tokenizer.save_pretrained(os.path.join(training_args.output_dir, f"epoch_{epoch+1}", "query_encoder"))
            context_tokenizer.save_pretrained(os.path.join(training_args.output_dir, f"epoch_{epoch+1}", "context_encoder"))
            query_model.module.save_pretrained(os.path.join(training_args.output_dir, f"epoch_{epoch+1}", "query_encoder"))
            context_model.module.save_pretrained(os.path.join(training_args.output_dir, f"epoch_{epoch+1}", "context_encoder"))


    