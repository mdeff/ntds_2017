"""
Plot function module for basic facts.
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import matplotlib as mpl

bl = '#58ACFA'

def plot_attack_ferq_per_year(attacks):
	plt.subplots(figsize=(15,6))
	sns.countplot(attacks['Year'], color=bl)
	plt.xticks(rotation=90)
	plt.title('Number of terrorist attacks per year')
	plt.ylabel('Frequency\n')
	plt.xlabel('\n Year')
	plt.show()

def plot_attack_types_freq(attacks):
	fig, axes = plt.subplots(figsize=(15,6))
	sns.countplot(attacks['Attack_type'],order=attacks['Attack_type'].value_counts().index,  color=bl)
	plt.xticks(fontsize=12, rotation=90)
	plt.title('Attacking methods')
	plt.ylabel('Frequency\n')
	plt.xlabel('\n Attack type ')
	fig.autofmt_xdate()
	plt.show()

def plot_target_distribution(attacks):
	fig, axes = plt.subplots(figsize=(15,6))
	sns.countplot(attacks['Target_type'],order=attacks['Target_type'].value_counts().index,  color=bl)
	plt.xticks(fontsize=12, rotation=90)
	plt.title('Targets distribution')
	plt.ylabel('Frequency\n')
	plt.xlabel('\n Target type')
	fig.autofmt_xdate()
	plt.show()

def plot_attack_freq_by_year_and_region(attacks):
	terror_region=pd.crosstab(attacks['Year'],attacks['Region'])
	axes = plt.subplot(111)
	axes.set_prop_cycle('color',plt.cm.spectral(np.linspace(0,1,12)))
	terror_region.plot(ax = axes)
	fig=plt.gcf()
	fig.set_size_inches(18,15)
	plt.title('Terrorist activity by year and region')
	plt.ylabel('Frequency \n')
	plt.xlabel('\n Year')
	plt.legend(title='Legend')
	plt.savefig('terrorist_attacks_by_year_and_region.pdf')
	plt.show()

def plot_attack_distribution_by_region(attacks):
	x = pd.crosstab(attacks['Region'],attacks['Attack_type'])
	axes = plt.subplot(111)
	x.plot(kind='barh', color = plt.cm.spectral(np.linspace(0,1,9)), ax=axes, stacked=True, width=1)
	fig=plt.gcf()
	fig.set_size_inches(12,8)
	plt.title('Distribution of attack types by region')
	plt.ylabel('Region \n')
	plt.xlabel('\n Frequency')
	plt.legend(title='Legend')
	plt.show()

def plot_most_affected_countries(attacks):
	fig, axes = plt.subplots(figsize=(18,6))
	sns.barplot(attacks['Country'].value_counts()[:20].index,attacks['Country'].value_counts()[:20].values, color=bl)
	plt.title('Most affected countries')
	plt.xticks(rotation=90)
	plt.ylabel('Frequency \n')
	plt.xlabel('\n Country')
	fig.autofmt_xdate()
	plt.savefig('most_affected_countries.pdf')
	plt.show()

def plot_top15_most_active_terrorist_groups(attacks):
	sns.barplot(attacks['Group'].value_counts()[1:15].values,attacks['Group'].value_counts()[1:15].index, color=bl)
	plt.xticks(rotation=90)
	fig=plt.gcf()
	fig.set_size_inches(10,8)
	plt.title('Top 15 of most active terrorist groups')
	plt.ylabel('Terrorist group \n')
	plt.xlabel('\n Frequency')
	plt.savefig('top15_active_groups.pdf')
	plt.show()

def joint_plot_coordinates(attacks):
	#usefull variable to get the coordinates.
	df_coords = attacks.round({'Longitude':0, 'Latitude':0}).groupby(["Longitude", "Latitude"]).size().to_frame(name = 'count').reset_index()
	fig=plt.gcf()
	fig.set_size_inches(10,8)
	sns.jointplot(x='Longitude', y='Latitude', data=df_coords, kind="kde", color=bl, size=15, stat_func=None, edgecolor="#020000", linewidth=.4)
	plt.savefig('joint_plot_coordinates.pdf')

def killed_num_attacks_relation(attacks):
	#compute the number of attacks for each country.
	number_atk_country = attacks['Country'].value_counts()
	#keeping the 'Country' and 'Killed' columns.
	killed_country = attacks[['Country','Killed']]
	#keeping the top 20.
	killed_country = killed_country.groupby(by='Country').sum().reset_index().sort_values(by='Killed',ascending=False)[:20]
	#getting the list.
	countries = list(killed_country['Country'])
	#taking the number of atack for each country in the list.
	countries_atk = [number_atk_country.loc[c] for c in countries]
	#creation of a new column.
	killed_country['Number attacks'] = countries_atk
	#let's look at the head.
	killed_country.head()
	#parameters.
	mpl.rcParams['figure.figsize'] = (15,5)
	mpl.rcParams['figure.dpi'] = 100
	#creation of an histo.
	histo = killed_country.plot.bar(color=['#FA5858', bl]);
	#setting x axis-
	histo.set_xticklabels(killed_country['Country'], rotation=45)
	#setting title.
	histo.set_title('Attacks and killed people per country')
	plt.ylabel('Frequency \n')
	plt.xlabel('\n Country')
	plt.tight_layout()
	plt.savefig('most_affected_countries_with_kills.pdf')
	plt.show();