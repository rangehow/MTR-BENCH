import os
import re
import hashlib
from typing import Any, Dict, List, Tuple, Set
from itertools import islice
import json
import multiprocessing as mp
import numpy as np
from collections import defaultdict
import datasets
from tqdm import tqdm

# Constants
MAX_HASH = np.uint64((1 << 64) - 1)  # 64-bit maximum hash value as np.uint64
MERSENNE_PRIME = 2**61 - 1  # Mersenne prime for hash calculations

# Regular expression for splitting non-alphanumeric characters
NON_ALPHA = re.compile(r"[^A-Za-z_0-9]")

# Define number of bands and rows per band for LSH
NUM_BANDS = 32  # Number of bands
ROWS_PER_BAND = 8  # Rows per band (NUM_BANDS * ROWS_PER_BAND should equal num_perm)

# Ensure that NUM_BANDS * ROWS_PER_BAND equals num_perm
NUM_PERM = 256  # Example: 256 permutations
assert NUM_BANDS * ROWS_PER_BAND == NUM_PERM, "NUM_BANDS * ROWS_PER_BAND must equal NUM_PERM"

# Generate hash ranges for each band
HASH_RANGES = [
    (i * ROWS_PER_BAND, (i + 1) * ROWS_PER_BAND) for i in range(NUM_BANDS)
]

# Generate random permutation parameters with unsigned integers
def generate_permutations(num_perm: int) -> np.ndarray:
    np.random.seed(42)  # Set seed for reproducibility
    # np.random.randint does not support np.uint64 directly, so use workaround
    # Generate random numbers in smaller chunks and combine them
    a = np.random.randint(1, MERSENNE_PRIME, size=num_perm, dtype=np.int64).astype(np.uint64)
    b = np.random.randint(0, MERSENNE_PRIME, size=num_perm, dtype=np.int64).astype(np.uint64)
    return np.vstack([a, b])

PERMUTATIONS = generate_permutations(num_perm=NUM_PERM)  # Example permutation parameters

# Helper functions

def sha1_hash(data: bytes) -> int:
    """Compute SHA1 hash and return its integer value truncated to 64 bits."""
    full_hash = int(hashlib.sha1(data).hexdigest(), 16)
    return full_hash & ((1 << 64) - 1)  # Truncate to 64 bits

def ngrams(iterable, n: int):
    """Generate n-grams."""
    iters = [iter(iterable)] * n
    return zip(*iters)

# Embed function implementation

def embed_func(
    content: str,
    idx: int,
    *,
    num_perm: int,
    ngram_size: int,
    hashranges: List[Tuple[int, int]],
    permutations: np.ndarray,
) -> Dict[str, Any]:
    a, b = permutations  # Both a and b are np.uint64 arrays
    
    # Generate n-gram tokens
    tokens: Set[str] = {" ".join(t) for t in ngrams(NON_ALPHA.split(content), ngram_size)}
    
    # Compute SHA1 hash values truncated to 64 bits
    hashvalues: np.ndarray = np.array([sha1_hash(token.encode("utf-8")) for token in tokens], dtype=np.uint64)
    
    if len(hashvalues) == 0:
        # Handle case with no tokens
        hashvalues = np.array([MAX_HASH], dtype=np.uint64)
    
    # Apply permutation functions
    # Ensure all operations are performed with np.uint64
    # Multiplication and addition with uint64 will remain within uint64
    permuted_hashvalues = np.bitwise_and(
        ((hashvalues[:, None] * a + b) % MERSENNE_PRIME),
        MAX_HASH
    )
    
    # Compute minimum hash values (MinHash signature)
    min_hashes = permuted_hashvalues.min(axis=0)
    
    # Split hash values into different bands (hash ranges)
    # Each band corresponds to a portion of the MinHash signature
    bands = [min_hashes[start:end] for start, end in hashranges]
    
    # Convert each band to bytes for hashing or storage
    band_hashes = [band.tobytes() for band in bands]
    
    return {"__signatures__": band_hashes, "__id__": idx}

# Example parameters (can be passed via command-line arguments or other means)
class Args:
    num_perm = NUM_PERM  # Number of hash permutations
    ngram = 5           # Size of n-grams

args = Args()

# Path to store and load the dataset
dataset_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/initial_clean"

# Load the dataset
ds = datasets.load_from_disk(dataset_path)['train']

# Apply the embed function in parallel using map
embedded = ds.map(
    function=embed_func,
    fn_kwargs={
        "num_perm": args.num_perm,
        "hashranges": HASH_RANGES,
        "ngram_size": args.ngram,
        "permutations": PERMUTATIONS,
    },
    input_columns='text',
    remove_columns=[col for col in ds.column_names if col != 'text'],  # 保留 'text' 列
    num_proc=os.cpu_count(),
    with_indices=True,
    desc="Generating LSH signatures...",
)

# 创建文档ID到文本的映射
id_to_text = {item['__id__']: ds[int(item['__id__'])]['text'] for item in embedded}

# Initialize a dictionary to map band hashes to document IDs
# 使用 defaultdict 可自动处理新键的集合
bucket_dict = defaultdict(set)

# Iterate over each document's signatures (bands)
for item in tqdm(embedded, desc='Assigning documents to LSH buckets'):
    band_hashes = item['__signatures__']
    doc_id = item['__id__']
    
    # Assign document ID to each band's bucket
    for band_hash in band_hashes:
        bucket_dict[band_hash].add(doc_id)

# Initialize a set to store duplicate document IDs
duplicates = set()
duplicates_to_save = []

# Identify duplicates by finding documents that share any bucket
for band_hash, doc_ids in bucket_dict.items():
    if len(doc_ids) > 1:
        duplicates_to_save.append(doc_ids)
        duplicates.update(doc_ids)

# 创建一个目录用于保存文件，如果该目录不存在的话
output_dir = "duplicates_files"
os.makedirs(output_dir, exist_ok=True)

# 遍历 duplicates_to_save 列表
for i, dup_set in enumerate(duplicates_to_save):
    # 将集合转换为列表，并获取对应的文本内容
    dup_list = [{"id": doc_id, "text": id_to_text[doc_id]} for doc_id in dup_set]
    
    # 定义文件名，例如：duplicates_0.json, duplicates_1.json,...
    file_name = f"duplicates_{i}.json"
    file_path = os.path.join(output_dir, file_name)

    # 将每组重复文档保存到一个独特的 JSON 文件中
    with open(file_path, "w", encoding='utf-8') as file:
        json.dump(dup_list, file, ensure_ascii=False, indent=4)
    
    print(f"Saved duplicates to {file_path}")

# remove the original dataset's indices corresponding to duplicates
unique_ds = ds.filter(
    lambda example, idx: idx not in duplicates,
    with_indices=True,
    desc="Filtering duplicates using LSH...",
    num_proc=os.cpu_count(),
)

# Save the filtered dataset if needed
unique_ds.save_to_disk("/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/ruanjunhao04/ruanjunhao/chatrag-bench/dedup")

print(f"Total documents: {len(ds)}")
print(f"Duplicates found: {len(duplicates)}")
print(f"Unique documents: {len(unique_ds)}")
