"""
Clustering Module.
"""

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
import pandas as pd


def best_k(index_min, index_max, data): 
	scaler = StandardScaler()
	data_scaled = scaler.fit_transform(data)
	cluster_range = range(index_min,index_max)
	cluster_errors = []

	for k in cluster_range:
	    clusters = KMeans(k)
	    clusters.fit(data_scaled)
	    cluster_errors.append(clusters.inertia_)

	clusters_df = pd.DataFrame( { "num_clusters": cluster_range, "cluster_errors": cluster_errors })
	plt.figure(figsize=(12,6))
	plt.plot( clusters_df.num_clusters, clusters_df.cluster_errors, marker = "o" )
	plt.show()


def best_k_gmm(index_min , index_max, data): 
	n_components = np.arange(index_min, index_max)
	models = [GMM(n, covariance_type='full', random_state=0).fit(data)
          for n in n_components]
    plt.plot(n_components, [m.bic(data) for m in models], label='BIC')
	plt.legend(loc='best')
	plt.xlabel('n_components');
	plt.show()


def prediction_kmm(n , data): 
	classifier = GMM(5, covariance_type='full', random_state=0).fit(data)
	result = classifier.predict(X = data)
	return result

def plot_prediction(label, prediction): 
	label = [x for x in label]
    Result = pd.DataFrame({
        "Label": label, 
        "Prediction": prediction, 
    })
    fig = plt.figure(figsize=(14, 6)) 

    sns.countplot(x = 'Prediction', hue = 'Label', data = Result)
    plt.title('Number of each ')
    plt.show()