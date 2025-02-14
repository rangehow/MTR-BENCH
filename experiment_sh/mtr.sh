cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench
export NUMEXPR_MAX_THREADS=1000


# DOCCHAT
# python eval.py --eval-dataset mtr --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --mtr-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/mtr_embedding/all_embeddings_Dragon-DocChat-Context-Encoder.npy --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/cerebras/Dragon-DocChat-Query-Encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/cerebras/Dragon-DocChat-Context-Encoder/main

wait

# dragon+
python eval.py --eval-dataset mtr --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --mtr-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/mtr_embedding/all_embeddings_dragon-plus-context-encoder.npy --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-query-encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-context-encoder/main

wait
# bge-ours
# python eval.py --eval-dataset mtr --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --mtr-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/mtr_embedding/all_embeddings_bge-large-8192.npy --query-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-best/checkpoint-624 --context-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-best/checkpoint-624
wait
# bge-syn
# python eval.py --eval-dataset mtr --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --mtr-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/mtr_embedding/all_embeddings_bge-synthethisqa.npy --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/abu1111/bge-synthesisQA-query-encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/abu1111/bge-synthesisQA-context-encoder/main
wait
# dragon-multi
# python eval.py --eval-dataset mtr --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --mtr-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/mtr_embedding/all_embeddings_dragon-multi.npy --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/nvidia/dragon-multiturn-query-encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/nvidia/dragon-multiturn-context-encoder/main
wait
# bge-large
python eval.py --eval-dataset mtr --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --mtr-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/mtr_embedding/all_embeddings_bge-large.npy --query-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/models/bge-large-en-v1.5 --context-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/models/bge-large-en-v1.5
