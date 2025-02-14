import json
import os
import datasets
import numpy as np
import faiss


def get_query(messages, num_turns=5):
    ## convert query into a format as follows:
    ## user: {user}\nagent: {agent}\nuser: {user}
    query = ""
    for item in messages[-num_turns:]:
        item['role'] = item['role'].replace("assistant", "agent")
        query += "{}: {}\n".format(item['role'], item['content'])
    query = query.strip()
    
    return query

def get_query_single(messages, num_turns=5):
    ## convert query into a format as follows:
    ## user: {user}\nagent: {agent}\nuser: {user}

    query = messages[-1]['content']
    
    
    return query



def get_query_with_topic(messages, topic, num_turns=3):
    ## convert query into a format as follows:
    ## user: this is a question about {topic}. {user}\nagent: {agent}\nuser: this is a question about {topic}. {user}
    query = ""
    for item in messages[-num_turns:]:
        item['role'] = item['role'].replace("assistant", "agent")
        if item['role'] == 'user':
            query += "{}: this is a question about {}. {}\n".format(item['role'], topic, item['content'])
        else:
            query += "{}: {}\n".format(item['role'], item['content'])
    query = query.strip()

    return query





ngpus=faiss.get_num_gpus()
gpu_resources = []
for i in range(ngpus):
    res = faiss.StandardGpuResources()
    gpu_resources.append(res)



def make_vres_vdev(ngpus, gpu_resources, i0=0, i1=-1):
    "Return vectors of device ids and resources useful for gpu_multiple"
    vres = faiss.GpuResourcesVector()
    vdev = faiss.Int32Vector()
    if i1 == -1:
        i1 = ngpus
    for i in range(i0, i1):
        vdev.push_back(i)
        vres.push_back(gpu_resources[i])
    return vres, vdev

def move_index_to_gpu(index: faiss.Index) -> faiss.Index:
    """
    将FAISS索引移动到GPU（如果可用）
    """
    if ngpus > 0:
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        vres, vdev = make_vres_vdev(ngpus, gpu_resources)
        gpu_index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)
        print("FAISS索引已成功移动到GPU。")
        return gpu_index
    else:
        print("没有可用的GPU，使用CPU索引。")
        return index






def get_data_for_evaluation_huge_dataset_rewrite(input_datapath, document_datapath, dataset_name):
    print('reading evaluation data from %s' % input_datapath)
    with open(input_datapath, "r") as f:
        input_list = json.load(f)
    
    eval_data = []
    for item in input_list:
        """
        We incorporate topic information for topiocqa and inscit datasets:
        
        """
        # import pdb
        # pdb.set_trace()
        # need to change this BUG
        
        query = "user: this is a question about {}. {}".format(item['topic'], item['rewrite_query'])
        

        gold_idx = item['ground_truth_ctx']['index']
        eval_data.append({"query": query, "gold_idx": gold_idx})

    

    if document_datapath.endswith('.npy'):
        dir_path = os.path.dirname(os.path.abspath(document_datapath))  # 使用绝对路径
        file_name = os.path.basename(document_datapath)
        index_save_path = os.path.join(dir_path, f"{file_name}.faiss")
        
        if os.path.exists(index_save_path):
            print("加载已有索引",index_save_path)
            index = faiss.read_index(index_save_path)
            index = move_index_to_gpu(index)
        else:
            try:
                # 加载npy文件
                print("读取嵌入")
                data = np.load(document_datapath)
                
                # 检查数据是否为二维数组（faiss需要的是二维数组，每行是一个向量）
                if data.ndim != 2:
                    raise ValueError(f"npy文件 {document_datapath} 的数据不是二维数组")
                
                # 获取数据的维度
                dimension = data.shape[1]

                # # 设置线程数（根据实际情况调整）
                # faiss.omp_set_num_threads(64) 
                
                # 创建FAISS索引并添加数据
                print("创建索引")
                index = faiss.IndexFlatIP(dimension)
                
                # index = faiss.index_factory(dimension, 'IVF32768,Flat', faiss.METRIC_INNER_PRODUCT)  # 根据数据集大小选择合适的索引类型
                # print("索引训练数据")
                # index.train(data)
                
                print("往索引添加数据")
                index.add(data)

                # 保存索引
                print("保存索引")
                faiss.write_index(index, index_save_path)
                index = move_index_to_gpu(index)
                
            except Exception as e:
                print(f"处理文件 {document_datapath} 时发生错误: {e}")
                raise
        

    return eval_data, index








def get_data_for_evaluation_huge_dataset(input_datapath, document_datapath, dataset_name):
    print('reading evaluation data from %s' % input_datapath)
    with open(input_datapath, "r") as f:
        input_list = json.load(f)
    
    eval_data = []
    for item in input_list:
        """
        We incorporate topic information for topiocqa and inscit datasets:
        
        """
        # import pdb
        # pdb.set_trace()
        query = get_query_with_topic(item['messages'], item['topic'])


        gold_idx = item['ground_truth_ctx']['index']
        eval_data.append({"query": query, "gold_idx": gold_idx})

    

    if document_datapath.endswith('.npy'):
        dir_path = os.path.dirname(os.path.abspath(document_datapath))  # 使用绝对路径
        file_name = os.path.basename(document_datapath)
        index_save_path = os.path.join(dir_path, f"{file_name}.faiss")
        
        if os.path.exists(index_save_path):
            print("加载已有索引",index_save_path)
            index = faiss.read_index(index_save_path)
            index = move_index_to_gpu(index)
        else:
            try:
                # 加载npy文件
                print("读取嵌入")
                data = np.load(document_datapath)
                
                # 检查数据是否为二维数组（faiss需要的是二维数组，每行是一个向量）
                if data.ndim != 2:
                    raise ValueError(f"npy文件 {document_datapath} 的数据不是二维数组")
                
                # 获取数据的维度
                dimension = data.shape[1]

                # # 设置线程数（根据实际情况调整）
                # faiss.omp_set_num_threads(64) 
                
                # 创建FAISS索引并添加数据
                print("创建索引")
                index = faiss.IndexFlatIP(dimension)
                
                # index = faiss.index_factory(dimension, 'IVF32768,Flat', faiss.METRIC_INNER_PRODUCT)  # 根据数据集大小选择合适的索引类型
                # print("索引训练数据")
                # index.train(data)
                
                print("往索引添加数据")
                index.add(data)

                # 保存索引
                print("保存索引")
                faiss.write_index(index, index_save_path)
                index = move_index_to_gpu(index)
                
            except Exception as e:
                print(f"处理文件 {document_datapath} 时发生错误: {e}")
                raise
        

    return eval_data, index


def get_data_for_evaluation_mtr(input_datapath, document_datapath):
    print('reading evaluation data from %s' % input_datapath)
    dataset = datasets.load_from_disk(input_datapath)
    
    eval_data = []
    for item in dataset:
        """
        We incorporate topic information for topiocqa and inscit datasets:
        
        """
        
        # query = get_query_with_topic(item['messages'], item['topic'])
        query = get_query(item['messages'],num_turns=16)
        
        # query = get_query_single(item['messages'])
        # query = get_query_with_topic(item['messages'], item['topic'])
        # query = item['messages']
        
        gold_idx = item['ground_truth_ctx']['index']
        eval_data.append({"query": query, "gold_idx": gold_idx})

    

    if document_datapath.endswith('.npy'):
        dir_path = os.path.dirname(os.path.abspath(document_datapath))  # 使用绝对路径
        file_name = os.path.basename(document_datapath)
        index_save_path = os.path.join(dir_path, f"{file_name}.faiss")
        
        if os.path.exists(index_save_path):
            print("加载已有索引",index_save_path)
            index = faiss.read_index(index_save_path)
            index = move_index_to_gpu(index)
        else:
            try:
                # 加载npy文件
                print("读取嵌入")
                data = np.load(document_datapath)
                
                # 检查数据是否为二维数组（faiss需要的是二维数组，每行是一个向量）
                if data.ndim != 2:
                    raise ValueError(f"npy文件 {document_datapath} 的数据不是二维数组")
                
                # 获取数据的维度
                dimension = data.shape[1]

                # # 设置线程数（根据实际情况调整）
                # faiss.omp_set_num_threads(64) 
                
                # 创建FAISS索引并添加数据
                print("创建索引")
                index = faiss.IndexFlatIP(dimension)
                
                # index = faiss.index_factory(dimension, 'IVF32768,Flat', faiss.METRIC_INNER_PRODUCT)  # 根据数据集大小选择合适的索引类型
                print("索引训练数据")
                # index.train(data)
                
                print("往索引添加数据")
                index.add(data)

                # 保存索引
                print("保存索引")
                faiss.write_index(index, index_save_path)
                index = move_index_to_gpu(index)
                
            except Exception as e:
                print(f"处理文件 {document_datapath} 时发生错误: {e}")
                raise
        

    return eval_data, index


def get_data_for_evaluation_mtr_rewrite(input_datapath, document_datapath):
    print('reading evaluation data from %s' % input_datapath)
    dataset = datasets.load_from_disk(input_datapath)
    
    eval_data = []
    for item in dataset:
        """
        We incorporate topic information for topiocqa and inscit datasets:
        
        """
        # import pdb
        # pdb.set_trace()
        # query = get_query_with_topic(item['messages'], item['topic'])
        # query = get_query(item['messages'],num_turns=16)
        
        query = item['rewrite_query']
        # query = get_query_with_topic(item['messages'], item['topic'])
        # query = item['messages']

        gold_idx = item['ground_truth_ctx']['index']
        eval_data.append({"query": query, "gold_idx": gold_idx})

    

    if document_datapath.endswith('.npy'):
        dir_path = os.path.dirname(os.path.abspath(document_datapath))  # 使用绝对路径
        file_name = os.path.basename(document_datapath)
        index_save_path = os.path.join(dir_path, f"{file_name}.faiss")
        
        if os.path.exists(index_save_path):
            print("加载已有索引",index_save_path)
            index = faiss.read_index(index_save_path)
            index = move_index_to_gpu(index)
        else:
            try:
                # 加载npy文件
                print("读取嵌入")
                data = np.load(document_datapath)
                
                # 检查数据是否为二维数组（faiss需要的是二维数组，每行是一个向量）
                if data.ndim != 2:
                    raise ValueError(f"npy文件 {document_datapath} 的数据不是二维数组")
                
                # 获取数据的维度
                dimension = data.shape[1]

                # # 设置线程数（根据实际情况调整）
                # faiss.omp_set_num_threads(64) 
                
                # 创建FAISS索引并添加数据
                print("创建索引")
                index = faiss.IndexFlatIP(dimension)
                
                # index = faiss.index_factory(dimension, 'IVF32768,Flat', faiss.METRIC_INNER_PRODUCT)  # 根据数据集大小选择合适的索引类型
                print("索引训练数据")
                # index.train(data)
                
                print("往索引添加数据")
                index.add(data)

                # 保存索引
                print("保存索引")
                faiss.write_index(index, index_save_path)
                index = move_index_to_gpu(index)
                
            except Exception as e:
                print(f"处理文件 {document_datapath} 时发生错误: {e}")
                raise
        

    return eval_data, index


def get_data_for_evaluation(input_datapath, document_datapath, dataset_name):

    print('reading evaluation data from %s' % input_datapath)
    with open(input_datapath, "r") as f:
        input_list = json.load(f)

    
    print('reading documents from %s' % document_datapath)
    with open(document_datapath, "r") as f:
        documents = json.load(f)


    eval_data = {}
    for item in input_list:
        """
        We incorporate topic information for topiocqa and inscit datasets:
        query = get_query_with_topic(item['messages'], item['topic'])
        """
        query = get_query(item['messages'])
        # query = get_query_single(item['messages'])
        # query = get_query_with_topic(item['messages'], item['topic'])
        # query = item['messages']

        doc_id = item['document']
        gold_idx = item['ground_truth_ctx']['index']

        if dataset_name == 'qrecc':
            """
            The 'gold context' for the qrecc dataset is obtained based on the word
            overlaps between gold answer and each context in the document, which might
            not be the real gold context.
            To improve the evaluation quality of this dataset,
            we further add the answer of the query into the 'gold context' 
            to ensure the 'gold context' is the most relevant chunk to the query.
            
            Note that this is just for the retrieval evaluation purpose, we do not
            add answer to the context for the ChatRAG evaluation.
            """
            answer = item['answers'][0]
            documents[doc_id][gold_idx] += " || "  + answer
        
        if doc_id not in eval_data:
            eval_data[doc_id] = [{"query": query, "gold_idx": gold_idx}]
        else:
            eval_data[doc_id].append({"query": query, "gold_idx": gold_idx})
        

    return eval_data, documents



def get_data_for_evaluation_rewrite(input_datapath, document_datapath, dataset_name):
    print("rewrite mode")
    print('reading evaluation data from %s' % input_datapath)
    with open(input_datapath, "r") as f:
        input_list = json.load(f)

    
    print('reading documents from %s' % document_datapath)
    with open(document_datapath, "r") as f:
        documents = json.load(f)


    eval_data = {}
    for item in input_list:
        """
        We incorporate topic information for topiocqa and inscit datasets:
        query = get_query_with_topic(item['messages'], item['topic'])
        """
        
        # query = get_query(item['messages'])
        # query = get_query_single(item['messages'])
        # query = get_query_with_topic(item['messages'], item['topic'])
        # query = item['messages']
        query = item['rewrite_query']

        doc_id = item['document']
        gold_idx = item['ground_truth_ctx']['index']

        if dataset_name == 'qrecc':
            """
            The 'gold context' for the qrecc dataset is obtained based on the word
            overlaps between gold answer and each context in the document, which might
            not be the real gold context.
            To improve the evaluation quality of this dataset,
            we further add the answer of the query into the 'gold context' 
            to ensure the 'gold context' is the most relevant chunk to the query.
            
            Note that this is just for the retrieval evaluation purpose, we do not
            add answer to the context for the ChatRAG evaluation.
            """
            answer = item['answers'][0]
            documents[doc_id][gold_idx] += " || "  + answer
        
        if doc_id not in eval_data:
            eval_data[doc_id] = [{"query": query, "gold_idx": gold_idx}]
        else:
            eval_data[doc_id].append({"query": query, "gold_idx": gold_idx})
        

    return eval_data, documents