#!/bin/bash

# 循环从1到9
for i in {1..10}
do
  # 执行 Python 脚本并传递参数 --start 和 --end
  python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/sglang_generate.py --start $i --end $i &
done
