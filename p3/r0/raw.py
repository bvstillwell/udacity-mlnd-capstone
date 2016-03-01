# Import libraries: NumPy, pandas, matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Tell iPython to include plots inline in the notebook
#%matplotlib inline

# Read dataset
data = pd.read_csv("wholesale-customers.csv")
print "Dataset has {} rows, {} columns".format(*data.shape)
print data.head()  # print the first 5 rows

for a in data:
    print '*' * 40
    print 'Feature :%12s' % a
    print 'Min     :%12.3f' % np.min(data[a])
    print 'Max     :%12.3f' % np.max(data[a])
    print 'Mean    :%12.3f' % np.mean(data[a])
    print 'Median  :%12.3f' % np.median(data[a])
    print 'Std     :%12.3f' % np.std(data[a], ddof=1) #ddof is for sample data
    print 'Var     :%12.3f' % np.var(data[a], ddof=1) #ddof is for sample data
    
# TODO: Apply PCA with the same number of dimensions as variables in the dataset
from sklearn.decomposition import PCA
pca = PCA(whiten=True)
pca.fit(data)

# Print the components and the amount of variance in the data contained in each dimension
print pca.components_
print pca.explained_variance_ratio_

plt.plot(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_, '-o')
plt.xlabel('dimension no')
plt.ylabel('standardised variance')
plt.title('Variance for dimension')
plt.xlim(xmin=0.5, xmax=6.5)
plt.grid = True
plt.show()

# TODO: Fit an ICA model to the data
# Note: Adjust the data to have center at the origin first!
from sklearn.decomposition import FastICA
ica = FastICA()
ica.fit(data)

# Print the independent components
print ica.components_

# Import clustering modules
from sklearn.cluster import KMeans
from sklearn.mixture import GMM

# TODO: First we reduce the data to two dimensions using PCA to capture variation
pca2 = PCA(n_components=2)
pca2.fit(data)
reduced_data = pca2.transform(data)
print reduced_data[:10]  # print upto 10 elements

# TODO: Implement your clustering algorithm here, and fit it to the reduced data for visualization
# The visualizer below assumes your clustering object is named 'clusters'
#clusters = KMeans(n_clusters=8)
clusters = GMM(n_components=3)
clusters.fit(reduced_data)
print clusters

# Plot the decision boundary by building a mesh grid to populate a graph.
x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
hx = (x_max-x_min)/1000.
hy = (y_max-y_min)/1000.
xx, yy = np.meshgrid(np.arange(x_min, x_max, hx), np.arange(y_min, y_max, hy))

# Obtain labels for each point in mesh. Use last trained model.
Z = clusters.predict(np.c_[xx.ravel(), yy.ravel()])

# TODO: Find the centroids for KMeans or the cluster means for GMM 

centroids = clusters.means_
print centroids

# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title('Clustering on the wholesale grocery dataset (PCA-reduced data)\n'
          'Centroids are marked with white cross')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()