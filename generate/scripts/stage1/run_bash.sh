#!/bin/bash

# 定义起始和结束值
start_value=21
end_value=50

# 循环从1到10
for ((i=start_value; i<=end_value; i++))
do
    # 调用你的脚本，传递当前的值作为参数
    bash sglang.sh $i $i
done