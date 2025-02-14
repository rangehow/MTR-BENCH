cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench
export NUMEXPR_MAX_THREADS=1000



# dragon+
python eval.py --eval-dataset doc2dial_rewrite --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --doc2dial-datapath doc2dial/test_rewrite.json --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-query-encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-context-encoder/main

wait

# bge-large
python eval.py --eval-dataset doc2dial_rewrite --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --doc2dial-datapath doc2dial/test_rewrite.json  --query-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/models/bge-large-en-v1.5 --context-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/models/bge-large-en-v1.5
