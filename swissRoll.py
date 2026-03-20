from LE import LE
from laplacianEmbedder import construct_adjacency_graph
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
n_samples = 2000
random_state = 2456

X, color = datasets.make_swiss_roll(n_samples=n_samples,random_state=random_state)

adjacencyGraph = construct_adjacency_graph(X,50, "knn")
#print(adjacencyGraph)

sns.heatmap(adjacencyGraph)
plt.show()

""" fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection = "3d")
ax.view_init(7,-80)
ax.scatter(X[:,0],X[:,1],X[:,2],c=color,cmap=plt.cm.jet)

plt.show()

le = LE(X, dim = 3, graph = 'eps', weights = 'heat kernel', 
        sigma = 5, laplacian = 'symmetrized')

Y = le.transform()

le.plot_embedding_2d(color, cmap=plt.cm.jet, grid = False, size = (14, 6)) """