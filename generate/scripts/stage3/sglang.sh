#!/bin/bash

# 检查是否传递了两个参数
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <start_value> <end_value>"
    exit 1
fi

# 获取位置参数
start=$1
end=$2

# 执行原始的命令
/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liuxinyu67/miniconda/bin/conda init
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/env/.bashrc
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/env/rjh
conda info
which conda
which python

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench

# 使用传递进来的 start 和 end 参数
python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/stage3.py --start "$start" --end "$end" --stage 6
