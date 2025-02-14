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
python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/generate/statistic/topic_classify.py