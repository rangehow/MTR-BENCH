from itertools import chain
import datasets
from langchain_text_splitters import RecursiveCharacterTextSplitter




def _clean(examples):
    cleaned=[]
    for example in examples:
        example=example.split('\n\nSee also')[0].split('\n\nReferences')[0].split('\n\nExternal links')[0].split('\n\nNotes and references')[0].split('\n\nFurther reading')[0]
        cleaned.append(example)

    return {"text":cleaned}

def _filter(examples):
    result=[]

    for example in examples:
        if len(example)>1024:
            result.append(True)
        else:
            result.append(False)
    return result



def group_texts(examples, max_length=1024):
    result = {key: [] for key in examples}
    text_splitter = RecursiveCharacterTextSplitter(
        separators=[
            "\n\n",
            "\n",
            " ",
            ".",
            ",",
            "\u200b",  # Zero-width space
            "\uff0c",  # Fullwidth comma
            "\u3001",  # Ideographic comma
            "\uff0e",  # Fullwidth full stop
            "\u3002",  # Ideographic full stop
            "",
        ],
        chunk_size=max_length,
        chunk_overlap=0,
        length_function=len,
        is_separator_regex=False,
    )


    for i in range(len(examples['text'])):
        text = examples['text'][i]
        sub_texts = text_splitter.split_text(text)
        
        


        # 处理其他的字段
        for sub_text in sub_texts:  # 忽略最后一个
            if len(sub_text)>=max_length//2:
                for key in examples:
                    if key == 'text':
                        result['text'].append(sub_text)
                    else:
                        result[key].append(examples[key][i])
            

    return result






dataset_path="/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/BERT_TRAINING_SERVICE/platform/dataset/wikimedia/wikipedia/main"



dataset = datasets.load_dataset(dataset_path,"20231101.en")


print(len(dataset['train']))
dataset = dataset.map(_clean,batched=True,input_columns='text',num_proc=100,load_from_cache_file=False)
dataset = dataset.filter(_filter,batched=True,input_columns='text',num_proc=100)
print(len(dataset['train']))

dataset = dataset['train'].map(group_texts, batched=True, num_proc=100,remove_columns='id')


dataset.save_to_disk("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean",num_proc=100)
