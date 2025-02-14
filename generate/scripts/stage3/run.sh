#!/bin/bash

# 设定文件夹路径（如果文件不在当前目录，需修改此路径）
folder_path="./"

# 循环执行sglang_4card_1.hope到sglang_4card_10.hope
for i in {1..100..2}
do
    # 构建文件名
    file="${folder_path}stage2_4card_${i}.hope"
    
    # 执行hope命令
    if [[ -f "$file" ]]; then
        echo "正在执行 $file"
        hope run "$file" 
    else
        echo "文件 $file 不存在"
    fi
done
