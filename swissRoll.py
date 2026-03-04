from LE import LE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets

n_samples = 2000
random_state = 42

X, color = datasets.make_swiss_roll(n_samples=n_samples,random_state=random_state)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111,projection = "3d")
ax.view_init(7,-80)
ax.scatter(X[:,0],X[:,1],X[:,2],c=color,cmap=plt.cm.jet)

plt.show()
