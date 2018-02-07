"""
Spectral Embedding module.
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.manifold import SpectralEmbedding


def se(features):
	"""
	Adds the two SE components to the features table.
	"""

	X = features.values
	se = SpectralEmbedding(n_components=2)
	Y = se.fit_transform(X)

	features['se1'] = Y[:, 0]
	features['se2'] = Y[:, 1]

	return features


def plot_killed_wounded_se(features):
	"""
	Plots the SE components for the 'Killed' and 'Wounded' fields of the features table.
	"""

	plt.figure(figsize=(20, 5))
	ax = plt.subplot(1, 2, 1)
	features.plot('se1', 'se2', kind='scatter', c='Killed', cmap='Reds', ax=ax)
	plt.ylabel('\n Second component')
	plt.xlabel('\n First component')
	plt.title('SE for number of killed people')
	ax = plt.subplot(1, 2, 2)
	features.plot('se1', 'se2', kind='scatter', c='Wounded', cmap='Blues', ax=ax)
	plt.ylabel('\n Second component')
	plt.xlabel('\n First component')
	plt.title('SE for number of wounded people')
	plt.tight_layout()
	plt.savefig('killed_wounded_se.pdf')
	plt.show()

def create_handles(dict_, ax):
	"""
	Creates the handles of the legend for the plot produced by the plot_attack_weapon_target_type function.
	"""
	
	points = []
	for key in dict_:
		points.append(matplotlib.axes.Axes.scatter(x=[], y=[], color=dict_[key], label=key, ax=ax))
    
	return points

def plot_attack_weapon_target_type(features, list_unique, list_):
	"""
	Plots the SE components for the 'Attack_type', 'Target_type' and 'Weapon_type' fields of the features table.
	"""

	# retrieve information
	list_attack_type_unique = list_unique[0]
	list_target_type_unique = list_unique[1]
	list_weapon_type_unique = list_unique[2]

	list_attack_type = list_[0]
	list_target_type = list_[1]
	list_weapon_type = list_[2]

	# we first define the colors we ill be using
	colors_attack_type = plt.cm.tab20(list(range(1,len(list_attack_type_unique)+1)))
	# we then create a dictionary which associates each value the attack_type can take with a corresponding color
	label_color_dict_attack_type = dict(zip(list_attack_type_unique, colors_attack_type))
	# we finally create our vector containing all the colors for each datapoint
	cvec_attack_type = [label_color_dict_attack_type[label] for label in list_attack_type]

	plt.figure(figsize=(10, 15))

	ax_1 = plt.subplot(3, 1, 1)
	plt.scatter(x=features['se1'], y=features['se2'], c=np.array(cvec_attack_type), edgecolor='')
	handles_1 = create_handles(label_color_dict_attack_type, ax_1)
	labels_1 = [h.get_label() for h in handles_1]
	plt.ylabel('\n Second component')
	plt.xlabel('\n First component')
	plt.title('SE for attack type')
	ax_1.legend(loc='upper left', bbox_to_anchor=(1, 1), handles=handles_1, labels=labels_1)

	# shorten a long legend for the plot
	list_weapon_type_unique[8] = 'Vehicule'
	list_weapon_type = ['Vehicule' if x=='Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)' else x for x in list_weapon_type]

	# same for weapon_type
	colors_weapon_type = plt.cm.tab20(list(range(1,len(list_weapon_type_unique)+1)))
	label_color_dict_weapon_type = dict(zip(list_weapon_type_unique, colors_weapon_type))
	cvec_weapon_type = [label_color_dict_weapon_type[label] for label in list_weapon_type]

	ax_2 = plt.subplot(3, 1, 2)
	plt.scatter(x=features['se1'], y=features['se2'], c=np.array(cvec_weapon_type), edgecolor='')
	handles_2 = create_handles(label_color_dict_weapon_type, ax_2)
	labels_2 = [h.get_label() for h in handles_2] 
	plt.ylabel('\n Second component')
	plt.xlabel('\n First component')
	plt.title('SE for weapon type')
	ax_2.legend(loc='upper left', bbox_to_anchor=(1, 1), handles=handles_2, labels=labels_2)  

	# same for target_type
	colors_target_type = plt.cm.tab20(list(range(1,len(list_target_type_unique)+1)))
	label_color_dict_target_type = dict(zip(list_target_type_unique, colors_target_type))
	cvec_target_type = [label_color_dict_target_type[label] for label in list_target_type]

	ax_3 = plt.subplot(3, 1, 3)
	plt.scatter(x=features['se1'], y=features['se2'], c=np.array(cvec_target_type), edgecolor='')
	handles_3 = create_handles(label_color_dict_target_type, ax_3)
	labels_3 = [h.get_label() for h in handles_3]
	plt.ylabel('\n Second component')
	plt.xlabel('\n First component')
	plt.title('SE for target type')
	ax_3.legend(loc='upper left', bbox_to_anchor=(1, 1), handles=handles_3, labels=labels_3)  

	plt.tight_layout()
	plt.savefig('attack_weapon_target_se.pdf', bbox_inches='tight')
	plt.show()