# Adds a csv file to the data directory of a specified dataset
# with relevance tags using that data directory's movie IDs

import pandas as pd
import numpy as np
from collections import defaultdict


base_dir = "./data/small/"


base_movies = pd.read_csv(base_dir + "movies.csv")
base_map = defaultdict(int, list(zip(base_movies.title, base_movies.movieId)))

# load in genome movie list
genome_movies = pd.read_csv("./data/genome/movies.dat", header=None, sep='\t')
genome_movies.columns = ["id", "title", "popularity"]
genome_map = defaultdict(int, list(zip(genome_movies.id, genome_movies.title)))




# Load in genome tag relevance

genome_relevance = pd.read_csv("./data/genome/tag_relevance.dat", header=None, sep='\t')
genome_relevance.columns = ["movieId", "tagId", "relevance"]





