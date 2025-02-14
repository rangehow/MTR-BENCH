from functools import partial
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import faiss
from torch.utils.data import Dataset, DataLoader

class MTRAG:
    def __init__(self, context_model_name_or_path, query_model_name_or_path=None) -> None:
        """对于双塔模型，会有两个encoder需要加载，对于单侧模型，query_model就是缺省的"""
        self.context_encoder = AutoModel.from_pretrained(context_model_name_or_path,torch_dtype="auto",trust_remote_code=True).cuda()
        if query_model_name_or_path is None:
            self.query_encoder = self.context_encoder
        else:
            self.query_encoder = AutoModel.from_pretrained(query_model_name_or_path,torch_dtype="auto",trust_remote_code=True).cuda()
        self.tokenizer = AutoTokenizer.from_pretrained(context_model_name_or_path) # ,truncation_side='left'
        self.index = None  # 初始化时不设置 index
        self.norm = None


    def fuseHistoryEmbeddings(self,*args):
        # return self.fuseHistoryEmbeddingsQuadratic(*args)
        return self.fuseHistoryEmbeddingsLinear(*args)

    def fuseHistoryEmbeddingsLinear(self, historyQueryEmbeddingList):
        
        if len(historyQueryEmbeddingList):
            historyQueryLen = len(historyQueryEmbeddingList)
            weights_for_sum = [x / sum(range(1, historyQueryLen + 1)) for x in range(1, historyQueryLen + 1)]
            return np.average(historyQueryEmbeddingList, weights=weights_for_sum, axis=0,keepdims=True)
        else:
            return None

    def fuseHistoryEmbeddingsQuadratic(self, historyQueryEmbeddingList):
        if len(historyQueryEmbeddingList):

            historyQueryLen = len(historyQueryEmbeddingList)
            # 后段平方加权，增加后段查询的权重
            weights_for_sum = [(i+1) ** 2 for i in range(historyQueryLen)]
            total_weight = sum(weights_for_sum)
            weights_for_sum = [w / total_weight for w in weights_for_sum]  # 归一化
            return np.average(historyQueryEmbeddingList, weights=weights_for_sum, axis=0, keepdims=True)
        
        else:
            return None

    def fuseHistoryEmbeddingsExponential(self, historyQueryEmbeddingList):
        # wocao
        if len(historyQueryEmbeddingList):
            historyQueryLen = len(historyQueryEmbeddingList)
            # 使用指数增长赋予后段更大的权重
            growth_factor = 1.2  # 增长因子
            weights_for_sum = [growth_factor ** i for i in range(historyQueryLen)]
            total_weight = sum(weights_for_sum)
            weights_for_sum = [w / total_weight for w in weights_for_sum]  # 归一化
            return np.average(historyQueryEmbeddingList, weights=weights_for_sum, axis=0, keepdims=True)
        else:
            return None

    


    def fuseHistoryEmbeddingsInverseExponential(self, historyQueryEmbeddingList):
        if len(historyQueryEmbeddingList):
            historyQueryLen = len(historyQueryEmbeddingList)
            # 反向指数加权，后段获得更大的权重
            decay_factor = 0.8  # 控制衰减速率
            weights_for_sum = [decay_factor ** (historyQueryLen - i - 1) for i in range(historyQueryLen)]
            total_weight = sum(weights_for_sum)
            weights_for_sum = [w / total_weight for w in weights_for_sum]  # 归一化
            return np.average(historyQueryEmbeddingList, weights=weights_for_sum, axis=0, keepdims=True)
        else:
            return None


    def fuseHistoryEmbeddingsCosine(self, historyQueryEmbeddingList):
        if len(historyQueryEmbeddingList):
            historyQueryLen = len(historyQueryEmbeddingList)
            # Handle the case where historyQueryLen is 1 to avoid division by zero
            if historyQueryLen == 1:
                return np.array(historyQueryEmbeddingList)  # Or handle as needed
            else:
                weights_for_sum = [0.5 * (1 + np.cos(np.pi * (i / (historyQueryLen - 1)))) for i in range(historyQueryLen)]
                return np.average(historyQueryEmbeddingList, weights=weights_for_sum, axis=0, keepdims=True)
        else:
            return None



    def fuseHistoryEmbeddingsGaussian(self, historyQueryEmbeddingList):
        if len(historyQueryEmbeddingList):
            historyQueryLen = len(historyQueryEmbeddingList)
            # 计算高斯分布的权重
            mean = (historyQueryLen - 1) / 2  # 中间位置
            stddev = historyQueryLen / 4  # 标准差
            weights_for_sum = [np.exp(-0.5 * ((i - mean) / stddev) ** 2) for i in range(historyQueryLen)]
            return np.average(historyQueryEmbeddingList, weights=weights_for_sum, axis=0, keepdims=True)
        else:
            return None
    

    

    def encode_queries(self,query,k=5,score_threshold=0.4,debug=True,border=[0,0],history_embedding_list=[]):
        
        
        # .count("\nagent:")
        if query.count("user: ")==1:
            # import pdb
            # pdb.set_trace()
            if debug:
                print("重置了")
            border=[0,0]
            history_embedding_list=[]

        
        query_embedding = self.encode_corpus([query],use_query_encoder=True)
        history_embedding_list.append(query_embedding[0])

        # import pdb
        # pdb.set_trace()
        # 退化的单轮搜索，要注意这个可能因为阈值返回空
        D,I = self.index.search(query_embedding,k=5*k) # 这个切片写法是为了keepdims
        D,I=D[0],I[0]


        # 先假定当前轮应该做融合,且和历史块是同义块
        border[-1]+=1
        historyQueryEmbeddingList=history_embedding_list[border[-2]:border[-1]]
        # 融合历史块
        history_query_embedding=self.fuseHistoryEmbeddings(historyQueryEmbeddingList)
        
        
        history_slice = history_embedding_list[border[-2]:border[-1]-2]
        if len(history_slice) > 0:
            EmbeddingListForSkipPrevTurnEmbedding = np.concatenate((history_slice, query_embedding), axis=0)
        else:
            EmbeddingListForSkipPrevTurnEmbedding = query_embedding

        SkipPrevTurnEmbedding=self.fuseHistoryEmbeddings(EmbeddingListForSkipPrevTurnEmbedding)
        
        if debug:
            print('调试，在融合前的同义块划分',border)
        
        # (history_query_embedding+query_embeddings[-1:,:])/2
        D_multi,I_multi, = self.index.search(history_query_embedding,k=5*k)
        D_multi,I_multi=D_multi[0],I_multi[0]
        
        # (SkipPrevTurnEmbedding+query_embeddings[-1:,:])/2
        # D_skip,I_skip, = self.index.search(SkipPrevTurnEmbedding,k=5*k)
        # D_skip,I_skip=D_skip[0],I_skip[0]

        if debug:
            print('调试，分数对比')
            print(f'单轮分数:{D[:5]},{sum(D[:k+1])},单轮结果{I[:20]}')
            print(f'多轮分数：{D_multi[:5]},{sum(D_multi[:k+1])},多轮结果{I_multi[:20]}')
            # print(f'跳跃分数：{D_skip[:5]},{sum(D_skip[:k+1])},多轮结果{I_skip[:20]}')

        # import pdb
        # pdb.set_trace()
        # 这里可能有个问题，就是单轮和多轮的搜索如果设置阈值，可能会返回空，使得结果不真实，因为列表是空就sum就是0
        # 但如果不设置阈值，可能要重新搜一次。\
        if len(D)<len(D_multi):
            if debug:
                print('调试','该问题单轮召回数目少于多轮，认定为不同块')
            # 先还原上面的假定
            border[-1]-=1
            border.append(border[-1]+1)
        elif sum(D[:k])>sum(D_multi[:k]): # 说明融合距离反而变远了，认定为非同义块
            if debug:
                print('调试','a非同一块')
            # 先还原上面的假定
            border[-1]-=1
            border.append(border[-1]+1)
        else:
            # 这里不需要做别的事情，因为上面已经假定为同义块
            if debug:
                print('调试','a是同一块')
            

        if debug:
            print('调试，融合后的同义块划分',border)
        


        # 一个大胆的尝试，当前轮的回答和同义块划分应当没有关系，直接把单多轮的结果混起来按照分数返回。
        # v2 其实混合应该是决定当前是单轮就不混多轮的，但是当前多轮可以考虑混单轮的？因为认定为单轮可以认为当前的信息比较丰富。？
        # 单轮一定要放在多轮前面，这决定了同分数的逻辑顺序，进而影响外面的直接返回
        single_total = list(zip(I, D, [0] * len(D)))
        mult_total = list(zip(I_multi, D_multi, [1] * len(D_multi)))
        total = single_total + mult_total


        # 按分数先排序一下，但是很可能有重复的，得筛！
        total=sorted(total,key=lambda x:x[1],reverse=True)
        docs=total
        # 5. 滤除检索结果里可能的重复项
        docs_set=set()
        valid_doc_count=0
        valid_doc=[]
        debug_use_for_source=[]
        for i,doc in enumerate(docs):
            if valid_doc_count>=k:
                break
            if doc[0] not in docs_set:
                docs_set.add(doc[0])
                valid_doc_count+=1
                # 这里doc[0]是Document，doc[1]是分数，doc[2]是来源于单论还是多轮的标签。
                # 我存分数是因为考虑到直接返回应该考虑分数。
                # valid_doc.append((doc[0],doc[1],doc[2]))
                valid_doc.append(doc[0])
                debug_use_for_source.append(doc[2])
        if debug:
            print('混合检索，目前是不稳定特性。单轮来源是0，多轮来源是1',debug_use_for_source)
        return valid_doc,border,history_embedding_list

    def initialize_index(self, texts, use_gpu=False, use_query_encoder=False):
        """初始化 FAISS 索引"""
        # print("开始编码文本并初始化索引...")
        embeddings = self.encode_corpus(texts, use_query_encoder=use_query_encoder)
        embedding_dim = embeddings.shape[-1]
        self.index = faiss.IndexFlatIP(embedding_dim)
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        self.index.add(embeddings)
        # print(f"索引初始化完成，包含 {self.index.ntotal} 个向量。")

    def encode_corpus(self, texts, use_query_encoder=False):
        class DocumentDataset(Dataset):
            def __init__(self, texts):
                self.texts = texts

            def __len__(self):
                return len(self.texts)

            def __getitem__(self, index):
                return self.texts[index]

        def collate_fn(batch, tokenizer):
            return tokenizer(batch, padding=True, max_length=512, truncation=True, return_tensors='pt')

        dataset = DocumentDataset(texts)
        dataloader = DataLoader(
            dataset, 
            batch_size=256, 
            shuffle=False, 
            num_workers=0,
            collate_fn=partial(collate_fn, tokenizer=self.tokenizer), 
            pin_memory=True
        )

        doc_embeddings = []
        encoder = self.query_encoder if use_query_encoder else self.context_encoder
        for encoded_input in dataloader:
            encoded_input=encoded_input.to(encoder.device)
            
            with torch.inference_mode():
    
                model_output = encoder(**encoded_input).last_hidden_state[:, 0, :]
                embeddings = model_output.cpu().numpy()
                doc_embeddings.append(embeddings)
        
        doc_embeddings = np.vstack(doc_embeddings)
        # if not use_query_encoder:
        #     norms = np.linalg.norm(doc_embeddings, axis=1)
        #     max_norm = norms.max()
        #     self.norm = max_norm

            
        #     doc_embeddings = doc_embeddings / max_norm


        return doc_embeddings

if __name__ == "__main__":
    texts = [" “The school motto 'Self-improvement and the Unity of Knowledge and Action' is inspiring and practical. It not only bears witness to the 90 years of legacy at Northeastern University but also reflects the diligent and down-to-earth spirit of the university community", "Feng Xiating, Ph.D. in engineering, professor, doctoral supervisor, and academician of the Chinese Academy of Engineering. He currently serves as the Deputy Party Secretary and President of Northeastern University", "On April 18, the Personnel Department of the Ministry of Education announced the appointment and removal decisions of the Party Leadership of the Ministry of Education at Southwest University. Comrade Wang Jinjun has been appointed President and Deputy Party Secretary of Southwest University.","The weather is beautiful today."]

    example_query = [
        {
            "document": "wiki",
            "messages": [
                {
                    "role": "user",
                    "content": "Do you know the school motto of Northeastern University?"
                },
                {
                    "role": "assistant",
                    "content": "回答1"
                },
                {
                    "role": "user",
                    "content": "Who is the president of the university?"
                }
            ],
            "answers": [
                "回答2"
            ],
            "ground_truth_ctx": {
                "index": 8,
                "ctx": "黄金文档内容"
            }
        }
    ]

    # 使用 context_encoder 和 query_encoder
    mtrag = MTRAG(
        "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-context-encoder/main",
        "/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/facebook/dragon-plus-query-encoder/main"
    )



    # 初始化索引，选择使用 context_encoder
    mtrag.initialize_index(texts, use_gpu=False, use_query_encoder=False)

    print(mtrag.encode_queries(example_query))

