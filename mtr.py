from transformers import AutoModel, AutoTokenizer
from dataset import get_data_for_evaluation,get_data_for_evaluation_huge_dataset,get_data_for_evaluation_mtr,get_data_for_evaluation_huge_dataset_rewrite,get_data_for_evaluation_rewrite
from arguments import get_args
from tqdm import tqdm
import torch
import os
from MTRAG_MT import MTRAG
import math

def run_retrieval(eval_data, documents, query_encoder_path, context_encoder_path, max_seq_len=512):
    ranked_indices_list = []
    gold_index_list = []
    for doc_id in tqdm(eval_data, desc="Running retrieval"):
        mtrag = MTRAG(context_encoder_path, query_encoder_path)
        context_list = documents[doc_id]
        mtrag.initialize_index(context_list)
      
        with torch.no_grad():
            # get chunk embeddings
            sample_list = eval_data[doc_id]
            border = [0, 0]
            history_embedding_list=[]
            for item in sample_list:
                gold_idx = item['gold_idx']
                gold_index_list.append(gold_idx)
                query = item['query']
                # 获取排名结果
                # print(query,gold_idx)
                result, border,history_embedding_list = mtrag.encode_queries(query, border=border, k=20, debug=False,history_embedding_list=history_embedding_list)
                # import pdb
                # pdb.set_trace()
                ranked_indices_list.append(result)
                # print("------------")
                

    return ranked_indices_list, gold_index_list







def run_retrieval_for_huge_dataset(eval_data, index, query_encoder_path, context_encoder_path, max_seq_len=512):
    print('huge')
    ranked_indices_list = []
    gold_index_list = []
    mtrag = MTRAG(context_encoder_path, query_encoder_path)
    mtrag.index=index
    border = [0, 0]
    history_embedding_list=[]
    with torch.no_grad():
        for item in tqdm(eval_data):
            
            gold_idx = item['gold_idx']
            gold_index_list.append(gold_idx)
            query = item['query']
            # 获取排名结果
            # print(query,gold_idx)
            result, border,history_embedding_list = mtrag.encode_queries(query, border=border, k=20, debug=False,history_embedding_list=history_embedding_list)
            
            ranked_indices_list.append(result)
            # print("------------")
            # import pdb
            # pdb.set_trace()

    return ranked_indices_list, gold_index_list




def calculate_recall(ranked_indices_list, gold_index_list, topk):
    hit = 0
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        if gold_index in ranked_indices[:topk]:
            hit += 1
    recall = hit / len(ranked_indices_list)
    print("Top-%d Recall: %.4f" % (topk, recall))

def calculate_mrr(ranked_indices_list, gold_index_list,k=20):
    reciprocal_rank_sum = 0.0
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        try:
            rank = ranked_indices.index(gold_index) + 1  # ranks start at 1
            if rank <= k:
                reciprocal_rank_sum += 1.0 / rank
            else:
                reciprocal_rank_sum += 0.0  # 超过k的位置不计入
        except ValueError:
            reciprocal_rank_sum += 0.0  # 未找到相关文档
    mrr_at_k = reciprocal_rank_sum / len(ranked_indices_list)
    print("Mean Reciprocal Rank (MRR@%d): %.4f" % (k, mrr_at_k))

def calculate_ndcg(ranked_indices_list, gold_index_list, k=20):
    dcg = 0.0
    idcg = 0.0
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        try:
            rank = ranked_indices.index(gold_index) + 1
            if rank <= k:
                dcg += 1.0 / math.log2(rank + 1)
        except ValueError:
            dcg += 0.0
        # 理想情况下，相关文档排名在第一位
        idcg += 1.0 / math.log2(1 + 1) if k >= 1 else 0.0
    ndcg = dcg / idcg if idcg > 0 else 0.0
    print("Normalized Discounted Cumulative Gain (NDCG@%d): %.4f" % (k, ndcg))

huge_dataset=['topiocqa','inscit']
def main():
    args = get_args()
    print('context_encoder_path',args.context_encoder_path)
    print('eval_dataset',args.eval_dataset)
    ## 获取评估数据路径
    if args.eval_dataset == "doc2dial" or args.eval_dataset == "doc2dial_rewrite":
        input_datapath = os.path.join(args.data_folder, args.doc2dial_datapath)
        input_docpath = os.path.join(args.data_folder, args.doc2dial_docpath)
    elif args.eval_dataset == "quac":
        input_datapath = os.path.join(args.data_folder, args.quac_datapath)
        input_docpath = os.path.join(args.data_folder, args.quac_docpath)
    elif args.eval_dataset == "qrecc":
        input_datapath = os.path.join(args.data_folder, args.qrecc_datapath)
        input_docpath = os.path.join(args.data_folder, args.qrecc_docpath)
    elif args.eval_dataset in huge_dataset :
        input_datapath = os.path.join(args.data_folder, getattr(args, f"{args.eval_dataset}_datapath"))
        input_docpath = getattr(args, f"{args.eval_dataset}_docpath", None)
        if input_docpath is None:
            raise ValueError(f"{args.eval_dataset}_docpath not found in args")
    elif args.eval_dataset=='topiocqa_rewrite':
        input_datapath = os.path.join(args.data_folder, getattr(args, f"topiocqa_datapath"))
        input_docpath = getattr(args, f"topiocqa_docpath", None)

    elif args.eval_dataset == 'mtr':
        input_datapath = getattr(args, f"{args.eval_dataset}_datapath")
        input_docpath = getattr(args, f"{args.eval_dataset}_docpath", None)
    else:
        raise Exception("请输入正确的 eval_dataset 名称！")

    # eval_data, documents = get_data_for_evaluation(input_datapath, input_docpath, args.eval_dataset)
    if  args.eval_dataset in huge_dataset:
        eval_data, index = get_data_for_evaluation_huge_dataset(input_datapath, input_docpath, args.eval_dataset)

        ranked_indices_list, gold_index_list = run_retrieval_for_huge_dataset(eval_data, index, args.query_encoder_path,
        args.context_encoder_path)
    
    elif args.eval_dataset == 'mtr':
        
        eval_data,index=get_data_for_evaluation_mtr(input_datapath, input_docpath)
        ranked_indices_list, gold_index_list = run_retrieval_for_huge_dataset(eval_data, index, args.query_encoder_path,
        args.context_encoder_path)

    elif args.eval_dataset == "doc2dial_rewrite":
        eval_data, documents = get_data_for_evaluation_rewrite(input_datapath, input_docpath, args.eval_dataset)
        ranked_indices_list, gold_index_list = run_retrieval(eval_data, documents, args.query_encoder_path,
        args.context_encoder_path)
    
    elif args.eval_dataset == "topiocqa_rewrite":
        eval_data, index = get_data_for_evaluation_huge_dataset_rewrite(input_datapath, input_docpath, args.eval_dataset)

        ranked_indices_list, gold_index_list = run_retrieval_for_huge_dataset(eval_data, index, args.query_encoder_path,
        args.context_encoder_path)

    else:
        eval_data, documents = get_data_for_evaluation(input_datapath, input_docpath, args.eval_dataset)
        ranked_indices_list, gold_index_list = run_retrieval(eval_data, documents, args.query_encoder_path,
        args.context_encoder_path)



    ## 运行检索
    # ranked_indices_list, gold_index_list = run_retrieval(
    #     eval_data,
    #     documents,
        
    # )
    print("总测试样本数: %d" % len(ranked_indices_list))

    ## 计算评估指标
    print("正在评估数据集: %s" % args.eval_dataset)
    
    # 计算 Recall@K
    topk_list = [1, 5, 20]
    for topk in topk_list:
        calculate_recall(ranked_indices_list, gold_index_list, topk=topk)
    
    # 计算 MRR
    calculate_mrr(ranked_indices_list, gold_index_list)
    
    # 计算 NDCG@20
    calculate_ndcg(ranked_indices_list, gold_index_list, k=20)

if __name__ == "__main__":
    main()
