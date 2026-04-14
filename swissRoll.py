#!/usr/bin/env python

#  from LE import LE
from laplacianEmbedder import (
    construct_adjacency_graph,
    eigen_decomposition,
    get_projection_metrix,
    add_weights,
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

n_samples = 2000
random_state = 23 
# random_state = None


X, color = datasets.make_swiss_roll(n_samples=n_samples, random_state=random_state)

adjacencyGraph = construct_adjacency_graph(X, 50, "weighted")


values, vectors = eigen_decomposition(X, t=15.0, n_neighbors=5, method="weighted")
labels = [f"l{i}" for i in range(0, len(values))]

chosen_vector_1 = 1
chosen_vector_2 = 2
# Visualizing Eigen values
eigen_value_df = pd.DataFrame({"Energy Levels": values}, labels)
print(eigen_value_df)

plot_energies = False 
if plot_energies:
    sns.barplot(eigen_value_df, x=labels, y="Energy Levels")
    plt.show()  # claim: the eigen values are increasting which lines up with expectation

# Creating plot
sns.scatterplot(
    x=vectors[:, chosen_vector_1],
    y=vectors[:, chosen_vector_2],
    c=color,
    palette="Spectral",
)
plt.show()

# # Side by side plot
# fig = plt.figure(figsize=(12, 5))

# ax1 = fig.add_subplot(121, projection="3d")
# ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="Spectral_r", s=5)
# ax1.set_title("Original Swiss Roll")

# ax2 = fig.add_subplot(122)
# ax2.scatter(
#     vectors[:, chosen_vector_1],
#     vectors[:, chosen_vector_2],
#     c=color,
#     cmap="Spectral",
#     s=5,
# )
# ax2.set_title("Laplacian Eigenmaps Embedding")

# plt.show()

fig = plt.figure(figsize=(14, 10))

# Top left: original swiss roll
ax1 = fig.add_subplot(221, projection='3d')
ax1.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap="Spectral_r", s=5)
ax1.set_title("Original Swiss Roll")

# Top right: embedding
ax2 = fig.add_subplot(222)
ax2.scatter(vectors[:, 1], vectors[:, 2], c=color, cmap="Spectral_r", s=5)
ax2.set_title("Laplacian Eigenmaps Embedding")

# Bottom center: eigenvalues
ax3 = fig.add_subplot(212)
ax3.bar(range(len(values)), values)
ax3.set_xlabel("Index")
ax3.set_ylabel("Eigenvalue")
ax3.set_title("Eigenvalue Spectrum")

plt.tight_layout()
plt.show()

# print(vectors.shape)
# projected_data = get_projection_metrix(X,vectors,2)
# print(projected_data)
# sns.scatterplot(x=projected_data[:,0],y=projected_data[:,1])
# plt.show()

# print(adjacencyGraph)

# sns.heatmap(add_weights(adjacencyGraph, t=5).toarray())
# plt.show()
