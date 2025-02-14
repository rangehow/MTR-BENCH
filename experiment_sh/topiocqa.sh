cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench



# docchat
# python eval.py --eval-dataset topiocqa --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --topiocqa-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embedding/all_embeddings_docchat.npy --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/cerebras/Dragon-DocChat-Query-Encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/cerebras/Dragon-DocChat-Context-Encoder/main

# dragon-multi
# python eval.py --eval-dataset topiocqa --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --topiocqa-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embedding/all_embeddings_dragon-multi.npy --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/nvidia/dragon-multiturn-query-encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/nvidia/dragon-multiturn-context-encoder/main

# dragon+
python eval.py --eval-dataset topiocqa --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --topiocqa-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embedding/all_embeddings_dragon_plus.npy --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-query-encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-context-encoder/main

# bge
python eval.py --eval-dataset topiocqa --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --topiocqa-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embedding/all_embeddings_bge-large.npy --query-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/models/bge-large-en-v1.5 --context-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/models/bge-large-en-v1.5


# bge-ours
# python eval.py --eval-dataset topiocqa --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --topiocqa-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embedding/all_embeddings_bge-ours.npy --query-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-best/checkpoint-624 --context-encoder-path /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-best/checkpoint-624

# bge-syn
# python eval.py --eval-dataset topiocqa --data-folder /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/dataset/ChatRAG-Bench/main/data --topiocqa-docpath /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/embedding/all_embeddings_bge-synthethisqa.npy --query-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/abu1111/bge-synthesisQA-query-encoder/main --context-encoder-path /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/abu1111/bge-synthesisQA-context-encoder/main
