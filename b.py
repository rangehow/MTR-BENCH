from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# construct prompt
question = "Who voices Darth Vader in Star Wars Episodes III-VI, IX Rogue One, and Rebels?"
heuristic_answer = "The voice of Darth Vader in Star Wars is provided by British actor James Earl Jones. He first voiced the character in the 1977 film \"Star Wars: Episode IV - A New Hope\", and his performance has been used in all subsequent Star Wars films, including the prequels and sequels."
prompt = (f"<s>[INST] <<SYS>>\nYou are a helpful assistant. Your task is to parse user input into"
          f" structured formats according to the coarse answer. Current datatime is 2023-12-20 9:47:28"
          f" <</SYS>>\n Course answer: (({heuristic_answer}))\nQuestion: (({question})) [/INST]")
params_query_rewrite = {"repetition_penalty": 1.05, "temperature": 0.01, "top_k": 1, "top_p": 0.85,
                        "max_new_tokens": 512, "do_sample": False, "seed": 2023}

# deploy model
model = AutoModelForCausalLM.from_pretrained("zstanjj/SlimPLM-Retrieval-Necessity-Judgment").eval()
if torch.cuda.is_available():
    model.cuda()
tokenizer = AutoTokenizer.from_pretrained("zstanjj/SlimPLM-Retrieval-Necessity-Judgment")

# run inference 
input_ids = tokenizer.encode(prompt.format(question=question, answer=heuristic_answer), return_tensors="pt")
len_input_ids = len(input_ids[0])
if torch.cuda.is_available():
    input_ids = input_ids.cuda()
outputs = model.generate(input_ids)
res = tokenizer.decode(outputs[0][len_input_ids:], skip_special_tokens=True)
print(res)
