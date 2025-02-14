/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liuxinyu67/miniconda/bin/conda init
source ~/.bashrc
conda activate /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/env/rjh
conda info
which conda
which python

cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/llm-sft-data/ruanjunhao/chatrag-bench/generate

export INPUT_FILE="original_data/non_fuhao_style_filtered_output2.json"
export OUTPUT_FILE="original_data/output2_results.jsonl"
export CACHE_DIR="cache"

python vllm_generate.py