#!/bin/bash

# 循环从 1 到 10
for i in {1..10}
do
    # 临时生成替换后的文件内容，输出到终端
    modified_content=$(sed "s/worker.script = bash sglang.sh [0-9]\+ [0-9]\+/worker.script = bash sglang.sh $i $i/" sglang_4card.hope)
    
    # 输出修改后的内容，供查看
    echo "Modified content for iteration $i:"
    echo "$modified_content"
    echo "--------------------------------------"
done
