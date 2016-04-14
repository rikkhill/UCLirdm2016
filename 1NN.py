import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
K = 10

movies = np.genfromtxt('output/dimmoviesK{}.csv'.format(K))
users = np.genfromtxt('output/dimusers{}.csv'.format(K)).transpose()

Tree = scipy.spatial.cKDTree(movies)

user_i = np.random.randint(users.shape[0])
user = users[user_i]

NN = Tree.query(user, 10)

movie_i = NN[1]
#movie = movies[movie_i]


for i in range(0, 10):
    print(movies[movie_i[i]])

"""
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1,K+1), user/np.linalg.norm(user))
ax.plot(range(1,K+1), movie/np.linalg.norm(movie))
fig.suptitle('user {}, movie {}'.format(user_i, movie_i))
plt.show(fig)
"""