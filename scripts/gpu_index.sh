
mpath=/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/liuxinyu67/miniconda/condabin/conda
${mpath} init     
source ~/.bashrc
conda activate knn
source /workdir/export_gid_index.sh


cd /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench


python /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/data_process/index.py --input_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/embeddings --output_dir /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/index --topk 64