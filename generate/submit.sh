# root.zw05_training_cluster.hadoop-aipnlp.llm_second
# root.zw05_training_cluster.hadoop-aipnlp.mm_pretrain
# root.zw06_training_cluster.hadoop-aipnlp.rl
# root.hldy_training_cluster.hadoop-aipnlp.h800_pool
# root.hldy_training_cluster.hadoop-aipnlp.h800_eval_debug

python submit.py \
    --config examples/configs/my_config_hl.json \
    --taskname mtrag \
    --workers 1 \
    --gpu 4 \
    --queue root.zw05_training_cluster.hadoop-aipnlp.llm_second \
    --usergroup hadoop-aipnlp \
    --run_env mt \
    --hope_tensorboard native \
    --image taming

# gpu需要加--image taming