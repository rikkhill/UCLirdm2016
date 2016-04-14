# Picks a random user and predicts movies for them

import numpy as np
import scipy.spatial
import pandas as pd
import matplotlib.pyplot as plt
import sys

try:
    sys.argv[1]
except IndexError:
    # Sensible default
    K = 10
else:
    K = int(sys.argv[1])

try:
    sys.argv[2]
except IndexError:
    N = 5
else:
    N = int(sys.argv[2])

base_dir = "./data/1M/"

movies = np.genfromtxt('output/dimmoviesK{}.csv'.format(K))
users = np.genfromtxt('output/dimusers{}.csv'.format(K)).transpose()
print(movies.shape)
print(users.shape)

movie_sums = np.linalg.norm(movies, axis=0)
movies = movies / movie_sums[np.newaxis, :]

user_sums = np.linalg.norm(users, axis=0)
users = users / user_sums[np.newaxis, :]

base_movies = pd.read_csv(base_dir + "movies.dat", sep="::", header=None, engine='python')
base_movies.columns = ["movieId", "title", "genre"]


Tree = scipy.spatial.cKDTree(movies)

user_i = np.random.randint(users.shape[0])
user = users[user_i]

NN = Tree.query(user, N)

movie_i = NN[1]


print("User ID: %d" % user_i)
print("Top bases:")
print(map(lambda x: x+1, users[user_i].argsort()[-3:][::-1]))

print("Recommended movies:")
for i in range(0, N):
    print("\t" + base_movies.iloc[movie_i[i]]["title"])


fig = plt.figure(figsize = (8, 6), dpi = 80)
ax = fig.add_subplot(211)
ax.plot(range(1,K+1), user, label = 'user {}'.format(user_i), linewidth=2, c='black') 
for i in movie_i:
    movie = movies[i] 
    ax.plot(range(1,K+1), movie, label = base_movies.iloc[i]["title"])
ax.legend(bbox_to_anchor=(0.5, -0.1), loc='upper center', ncol=1)
#this is to make the user to be on top, but the first on the legend
ax.plot(range(1,K+1), user, label = 'user {}'.format(user_i), linewidth=2, c='black') 
plt.savefig('plot.png', bbox_inches='tight')
plt.show(fig)