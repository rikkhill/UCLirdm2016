from pymf import NMF, WNMF, base
import numpy as np
import pandas as pd

df = pd.read_csv("./data/1M/ratings.dat", sep='::')
df.columns = ['userId', 'movieId', 'rating', 'timestamp']
ratings = df.pivot(index="movieId", columns="userId", values="rating")
ratings.fillna(0, inplace=True)

data = ratings.as_matrix()
(d_m, d_n) = data.shape

# Lets make a bi-cross-validation weight matrix
hold_out_proportion = 0.2
m_indices = np.random.choice(d_m, int(d_m * hold_out_proportion), replace=False).tolist()
n_indices = np.random.choice(d_n, int(d_n * hold_out_proportion), replace=False).tolist()

weight_matrix = np.ones(data.shape)
weight_matrix[np.array(m_indices)[:, None], n_indices] = 0


def callout(arg):
    print(arg.frobenius_norm(complement=True))

K = 10
wnmf = WNMF(data, weight_matrix, num_bases=K)
wnmf.factorize(niter=50, show_progress=True, epoch_hook=lambda x: callout(x))
