/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liuxinyu67/miniconda/bin/conda init
source /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/env/.bashrc
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/env/rjh
conda info
which conda
which python

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench


# python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/data_process/embedding.py --dataset /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/models/gte-Qwen2-7B-instruct/main --model_name gte-qwen-7b --output_dir  ./mtr_embedding


# python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/data_process/embedding.py --dataset /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean --model /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/nvidia/dragon-multiturn-context-encoder/main --model_name dragon-multi --output_dir  ./mtr_embedding


# python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/data_process/embedding.py --dataset /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean --model /mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/abu1111/bge-synthesisQA-context-encoder/main --model_name bge-synthethisqa --output_dir  ./mtr_embedding

# bge-best
# python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/data_process/embedding.py --dataset /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-acc64-8w/checkpoint-624 --model_name bge-best --output_dir  ./mtr_embedding

# bge-fuck

python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/data_process/embedding.py --dataset /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean --model /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-4w-fuck-acc64/checkpoint-156 --model_name bge-fuck --output_dir  ./mtr_embedding