#!/bin/bash



# 执行原始的命令
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liuxinyu67/miniconda/bin/conda init
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/env/.bashrc
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/env/rjh
conda info
which conda
which python

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench

# 使用传递进来的 start 和 end 参数
python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/statistic/quality_check.py --model_dir /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/mistralai/Mistral-Large-Instruct-2407/main

# /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/NousResearch/Meta-Llama-3.1-70B-Instruct/main
# /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main
# /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/Qwen/Qwen2-72B-Instruct/main
# /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/mistralai/Mistral-Large-Instruct-2407/main