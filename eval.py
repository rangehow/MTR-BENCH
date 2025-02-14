
from transformers import AutoModel, AutoTokenizer
from dataset import get_data_for_evaluation,get_data_for_evaluation_huge_dataset,get_data_for_evaluation_mtr,get_data_for_evaluation_huge_dataset_rewrite,get_data_for_evaluation_rewrite,get_data_for_evaluation_mtr_rewrite
from arguments import get_args
from tqdm import tqdm
import torch
import os
import math
import faiss


def run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer, max_seq_len=512, bsz=32):
    max_seq_len = query_encoder.config.max_position_embeddings
    print('max_seq_len', max_seq_len)
    
    ranked_indices_list = []
    gold_index_list = []
    
    # Process the documents in batches
    for doc_id in tqdm(eval_data):
        context_list = documents[doc_id]
        
        with torch.inference_mode():
            # Get chunk embeddings for the context in batch
            context_embs = []
            for i in range(0, len(context_list), bsz):  # Batch processing for context
                context_batch = context_list[i:i+bsz]
                context_ids = tokenizer(context_batch, max_length=max_seq_len, truncation=True, padding=True, return_tensors="pt").to("cuda")
                
                c_emb = context_encoder(input_ids=context_ids.input_ids, attention_mask=context_ids.attention_mask)
                c_emb = c_emb.last_hidden_state[:, 0, :]  # Take the [CLS] token embedding
                context_embs.append(c_emb)
            context_embs = torch.cat(context_embs, dim=0)   # (num_chunk, hidden_dim)
    
            sample_list = eval_data[doc_id]
            query_embs = []
            
            for i in range(0, len(sample_list), bsz):  # Batch processing for queries
                query_batch = sample_list[i:i+bsz]
                queries = [item['query'] for item in query_batch]
                query_ids = tokenizer(queries, max_length=max_seq_len, truncation=True, padding=True, return_tensors="pt").to("cuda")
                
                q_emb = query_encoder(input_ids=query_ids.input_ids, attention_mask=query_ids.attention_mask)
                q_emb = q_emb.last_hidden_state[:, 0, :]  # Take the [CLS] token embedding
                query_embs.append(q_emb)

                for item in query_batch:
                    gold_idx = item['gold_idx']
                    gold_index_list.append(gold_idx)
            
            query_embs = torch.cat(query_embs, dim=0)   # (num_query, hidden_dim)
            
            # Compute similarities in batch
            similarities = query_embs.matmul(context_embs.transpose(0, 1))  # (num_query, num_chunk)
            ranked_results = torch.argsort(similarities, dim=-1, descending=True)  # (num_query, num_chunk)
            ranked_indices_list.extend(ranked_results.tolist())

    return ranked_indices_list, gold_index_list







def run_retrieval_for_huge_dataset(eval_data, index, query_encoder,context_encoder,tokenizer, max_seq_len=512):

    
    ranked_indices_list = []
    gold_index_list = []
    query_embs = []
    for item in tqdm(eval_data):
        with torch.inference_mode():
            gold_idx = item['gold_idx']
            gold_index_list.append(gold_idx)

            query = item['query']
            query_ids = tokenizer(query, max_length=query_encoder.config.max_position_embeddings, truncation=True,return_tensors="pt").to("cuda")
            q_emb = query_encoder(input_ids=query_ids.input_ids, attention_mask=query_ids.attention_mask)
            q_emb = q_emb.last_hidden_state[:, 0, :]
            query_embs.append(q_emb)
            

    query_embs = torch.cat(query_embs, dim=0).cpu().numpy()   # (num_query, hidden_dim)
    
    _,I = index.search(query_embs,k=20)
    try:
        ranked_indices_list=I.tolist()

    except:
        import pdb
        pdb.set_trace()
    

    return ranked_indices_list, gold_index_list




def calculate_recall(ranked_indices_list, gold_index_list, topk):
    hit = 0
    
    for ranked_indices, gold_index in zip(ranked_indices_list, gold_index_list):
        
        for idx in ranked_indices[:topk]:
            try:
                if int(idx) == int(gold_index):
                    hit += 1
                    break
            except:
                import pdb
                pdb.set_trace()
    recall = hit / len(ranked_indices_list)

    print("top-%d recall score: %.4f" % (topk, recall))


def calculate_mrr(ranked_indices_list, gold_index_list, k=20):
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
    ## get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.query_encoder_path)

    ## get retriever model
    query_encoder = AutoModel.from_pretrained(args.query_encoder_path,trust_remote_code=True)
    context_encoder = AutoModel.from_pretrained(args.context_encoder_path,trust_remote_code=True)
    query_encoder.to("cuda"), query_encoder.eval()
    context_encoder.to("cuda"), context_encoder.eval()

    ## get evaluation data
    if args.eval_dataset == "doc2dial" or args.eval_dataset == "doc2dial_rewrite":
        input_datapath = os.path.join(args.data_folder, args.doc2dial_datapath)
        input_docpath = os.path.join(args.data_folder, args.doc2dial_docpath)

    elif args.eval_dataset == "quac" or args.eval_dataset == "quac_rewrite":
        input_datapath = os.path.join(args.data_folder, args.quac_datapath)
        input_docpath = os.path.join(args.data_folder, args.quac_docpath)
    elif args.eval_dataset == "qrecc" or args.eval_dataset == "qrecc_rewrite":
        input_datapath = os.path.join(args.data_folder, args.qrecc_datapath)
        input_docpath = os.path.join(args.data_folder, args.qrecc_docpath)
    elif args.eval_dataset in huge_dataset:
        input_datapath = os.path.join(args.data_folder, getattr(args, f"{args.eval_dataset}_datapath"))
        input_docpath = getattr(args, f"{args.eval_dataset}_docpath", None)
        if input_docpath is None:
            raise ValueError(f"{args.eval_dataset}_docpath not found in args")
    elif args.eval_dataset == 'mtr' or args.eval_dataset=='mtr_rewrite':
        input_datapath = getattr(args, f"mtr_datapath")
        input_docpath = getattr(args, f"mtr_docpath", None)
    elif args.eval_dataset=='topiocqa_rewrite':
        input_datapath = os.path.join(args.data_folder, getattr(args, f"topiocqa_datapath"))
        input_docpath = getattr(args, f"topiocqa_docpath", None)
    
    else:
        raise Exception("Please input a correct eval_dataset name!")

    if  args.eval_dataset in huge_dataset:
        
        eval_data, index = get_data_for_evaluation_huge_dataset(input_datapath, input_docpath, args.eval_dataset)
        ranked_indices_list, gold_index_list = run_retrieval_for_huge_dataset(eval_data, index, query_encoder,context_encoder, tokenizer)
    elif args.eval_dataset == 'mtr':
        
        eval_data,index=get_data_for_evaluation_mtr(input_datapath, input_docpath)
        ranked_indices_list, gold_index_list = run_retrieval_for_huge_dataset(eval_data, index, query_encoder,context_encoder, tokenizer)

    elif args.eval_dataset == 'mtr_rewrite':
        
        eval_data,index=get_data_for_evaluation_mtr_rewrite(input_datapath, input_docpath)
        ranked_indices_list, gold_index_list = run_retrieval_for_huge_dataset(eval_data, index, query_encoder,context_encoder, tokenizer)

    elif args.eval_dataset == "doc2dial_rewrite" or args.eval_dataset == "quac_rewrite" or args.eval_dataset == "qrecc_rewrite":
        eval_data, documents = get_data_for_evaluation_rewrite(input_datapath, input_docpath, args.eval_dataset)
        ranked_indices_list, gold_index_list = run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer)
    
    elif args.eval_dataset == "topiocqa_rewrite":
        eval_data, index = get_data_for_evaluation_huge_dataset_rewrite(input_datapath, input_docpath, args.eval_dataset)

        ranked_indices_list, gold_index_list = run_retrieval_for_huge_dataset(eval_data, index, query_encoder,context_encoder, tokenizer)
    else:
        eval_data, documents = get_data_for_evaluation(input_datapath, input_docpath, args.eval_dataset)
        ranked_indices_list, gold_index_list = run_retrieval(eval_data, documents, query_encoder, context_encoder, tokenizer)
    
    ## run retrieval
    
    import json
    json.dump(ranked_indices_list,open('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/a.json','w'),indent=2)
    print("number of the total test samples: %d" % len(ranked_indices_list))

    ## calculate recall scores
    print("evaluating on %s" % args.eval_dataset)
    
    topk_list = [1, 5, 20]
    for topk in topk_list:
        calculate_recall(ranked_indices_list, gold_index_list, topk=topk)

    # 计算 MRR
    calculate_mrr(ranked_indices_list, gold_index_list)
    
    # 计算 NDCG@20
    calculate_ndcg(ranked_indices_list, gold_index_list, k=20)
if __name__ == "__main__":
    main()