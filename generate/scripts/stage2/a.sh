#!/bin/bash

# 循环从 1 到 10
for i in {1..100}
do
    # 临时生成替换后的文件内容
    modified_content=$(sed "s/worker.script = bash sglang.sh [0-9]\+ [0-9]\+/worker.script = bash sglang.sh $i $i/" ../sglang_4card.hope)
    
    # 为每次迭代生成一个新的 .sh 文件
    output_file="stage2_4card_$i.hope"
    
    # 将修改后的内容写入新的文件
    echo "$modified_content" > "$output_file"
    
    # 输出提示信息
    echo "Generated $output_file"
done
