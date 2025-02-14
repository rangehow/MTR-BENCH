import json
from tqdm import tqdm
from fuzzywuzzy import fuzz


# 读取tsv文件并根据ctx建立一个字典映射
tsv_file = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/dataset/full_wiki_segments.tsv'

json_file = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/dataset/ChatRAG-Bench/main/data/topiocqa/dev.json'
output_file = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/dataset/ChatRAG-Bench/main/data/topiocqa/modified_dev.json'
title_to_text_to_id = {}






# 读取TSV文件并构建映射
with open(tsv_file, 'r', encoding='utf-8') as f:
    # 跳过表头
    next(f)
    for line in tqdm(f, desc="Processing TSV file", unit="line"):
        id, text, title = line.strip().split('\t')
        text = text.strip()

        # 如果title还没有在字典中，初始化空的字典
        if title not in title_to_text_to_id:
            title_to_text_to_id[title] = {}

        # 将text映射到id
        title_to_text_to_id[title][text] = id

# 读取JSON文件，逐条更新ground_truth_ctx

# 加载整个JSON文件
with open(json_file, 'r', encoding='utf-8') as f:
    data = json.load(f)  # 假设JSON文件是一个列表，包含多个对象

# 获取数据的总长度，用于显示进度条
total_entries = len(data)

# 遍历JSON数据，更新每个条目的ground_truth_ctx
for entry in tqdm(data, total=total_entries, desc="Processing JSON file", unit="entry"):
    ground_truth_ctx = entry.get('ground_truth_ctx', {})
    ctx = ground_truth_ctx.get('ctx').strip()
    title = ground_truth_ctx.get('title') + " [SEP] " + ground_truth_ctx.get('subtitle')

    # 模糊匹配找到最相似的text和对应的id
    best_match_score = 0
    best_match_id = None
    if title in title_to_text_to_id:
        for text, id in title_to_text_to_id[title].items():
            # 计算ctx与每个text之间的相似度
            score = fuzz.partial_ratio(ctx, text)  # 使用部分相似度，可以调整为其他类型的匹配
            if score > best_match_score:
                best_match_score = score
                best_match_id = id

    # 如果找到了最相似的text，添加index
    if best_match_id:
        ground_truth_ctx['index'] = int(best_match_id)-1
    else:
        # 如果没有找到对应的匹配，可以选择赋予一个特殊的值（例如None或-1），以便后续处理
        ground_truth_ctx['index'] = None  # 或者使用 -1 或其他默认值

    # 更新条目的ground_truth_ctx
    entry['ground_truth_ctx'] = ground_truth_ctx

# 将更新后的数据写入新文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False,)

print("修改完成，新的json文件已保存为 'modified_dev.json'")
