import numpy as np
from cuml.dask.cluster import DBSCAN
from dask.distributed import Client
from dask_cuda import LocalCUDACluster

if __name__ == "__main__":
    cluster = LocalCUDACluster()
    client = Client(cluster)
    embs = np.random.randn(100_000, 256)
    dbscan = DBSCAN(
        client=client,
        eps=0.25,
        min_samples=5,
        metric="cosine",
    ).fit(embs)