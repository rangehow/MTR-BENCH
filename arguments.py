
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser(description="Dragon-multiturn")

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-query-encoder/main')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-context-encoder/main')

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/dragon/epoch_2/query_encoder')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/dragon/epoch_2/context_encoder')

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192/checkpoint-2500')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192/checkpoint-2500')
    
    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-bad/checkpoint-624')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-bad/checkpoint-624')
    

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/dragon/query_encoder')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/dragon/context_encoder')
    

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-best/checkpoint-450')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192-best/checkpoint-450')

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192/checkpoint-1250')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192/checkpoint-1250')


    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/abu1111/bge-synthesisQA-query-encoder/main')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/abu1111/bge-synthesisQA-context-encoder/main')




    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large/checkpoint-450')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large/checkpoint-450')
    

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192/checkpoint-1250')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/train/model_dir/bge-large-8192/checkpoint-1250')

    
    parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/models/bge-large-en-v1.5')
    parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/models/bge-large-en-v1.5')

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/nvidia/dragon-multiturn-query-encoder/main')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/nvidia/dragon-multiturn-context-encoder/main')

    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/cerebras/Dragon-DocChat-Query-Encoder/main')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/cerebras/Dragon-DocChat-Context-Encoder/main')


    # parser.add_argument('--query-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/models/stella_en_400M_v5')
    # parser.add_argument('--context-encoder-path', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/models/stella_en_400M_v5')


    parser.add_argument('--data-folder', type=str, default='', help='path to the datafolder of ChatRAG Bench')
    parser.add_argument('--eval-dataset', type=str, default='', help='evaluation dataset (e.g., doc2dial)')

    parser.add_argument('--doc2dial-datapath', type=str, default='doc2dial/test.json')
    parser.add_argument('--doc2dial-docpath', type=str, default='doc2dial/documents.json')

    parser.add_argument('--quac-datapath', type=str, default='quac/test.json')
    parser.add_argument('--quac-docpath', type=str, default='quac/documents.json')
    
    parser.add_argument('--qrecc-datapath', type=str, default='qrecc/test.json')
    parser.add_argument('--qrecc-docpath', type=str, default='qrecc/documents.json')
    
    # argument that needed by pre index dataset =====================================
    parser.add_argument('--topiocqa-datapath', type=str, default='topiocqa/modified_dev.json')
    parser.add_argument('--topiocqa-docpath', type=str, default='')
    
    parser.add_argument('--inscit-datapath', type=str, default='inscit/dev.json')
    parser.add_argument('--inscit-docpath', type=str, default='')

    parser.add_argument('--mtr-datapath', type=str, default='/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/generate/mtr-test-paired-chatrag-format')
    parser.add_argument('--mtr-docpath', type=str, default='')
    # ================================================================================
    



    args = parser.parse_args()

    return args