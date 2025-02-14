import datasets
# import sglang as sgl
from tqdm import tqdm
from transformers import AutoTokenizer

classify_prompt="""Please classify the following document into the appropriate category. The categories include but are not limited to:

- Information Technology (IT)
- Finance
- Education
- Healthcare
- Energy
- Retail
- Manufacturing
- Agriculture
- Transportation and Logistics
- Government and Public Utilities
- Social Theory
- International Political Science
- Philosophy
- Law
- Psychology
- Sociology
- Anthropology
- Political Science
- Cultural Studies
- Environmental Science
- History
- Economics

Please read the document carefully and select the most appropriate category based on its content. Only return the category, no need for explanation. If the document covers multiple fields, choose the most relevant one. If the document is difficult to categorize, return "No Suitable Category."

Document content:
{document}
"""

if __name__=='__main__':
    dataset = datasets.load_from_disk('/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/INS/ruanjunhao04/ruanjunhao/chatrag-bench/cluster_wow/0')
    print(dataset[0])

    start=0
    end=0
    # model_path="/mnt/hdfs/zw04mlnn01/checkpoint/llm_platform/model/unsloth/Llama-3.3-70B-Instruct/main"
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset_list=[]
    for i in range(start,end+1):
        
        dataset = datasets.load_from_disk(f'/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/cluster_wow/{i}')
        import pdb;pdb.set_trace()
        dataset_list.append(dataset)

    dataset = datasets.concatenate_datasets(dataset_list)
    # dataset = dataset.map(partial(first_row_query,prompt=PROMPT,tokenizer=tokenizer),batched=True,num_proc=32,)
    # sampling_params = {"temperature": 0.8, "top_p": 0.95,'max_new_tokens':10000}
    # llm = sgl.Engine(model_path=model_path,tp_size=4,dp_size=2 if device_count()>4 else 1,context_length=128000)