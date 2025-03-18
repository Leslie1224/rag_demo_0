import numpy as np
d = 64                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss
nlist = 100
k = 4
quantizer = faiss.IndexFlatL2(d)  # 另外一个 Index
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
       # 这里我们指定了 METRIC_L2, 默认它执行 inner-product 搜索。
assert not index.is_trained
index.train(xb)
assert index.is_trained

index.add(xb)                  # add may be a bit slower as well
D, I = index.search(xq, k)     # actual search
#不设置index.nprobe的时候，默认index.nprobe = 1，快但准确性不保证
print(I[-5:])                  # neighbors of the 5 last queries

index.nprobe = 10              # default nprobe is 1, try a few more
D, I = index.search(xq, k)
print(I[-5:])                  # neighbors of the 5 last queries