"""
Time series module.
"""

import matplotlib.pyplot as plt
from datetime import datetime

def create_time_series(attacks):
	"""
	Creates a datetime row to simplify the time series analysis.
	"""
	ts = attacks
	# delete the rows where we don't know the month or the day
	ts = ts.drop(ts[(ts['Month'] == 0) | (ts['Day'] == 0)].index)
	# convert into datetime format
	ts['DateTime'] = ts[['Year', 'Month', 'Day']].apply(lambda s : datetime(*s),axis = 1)
	# drop unnecessary fields
	ts.drop(['Year', 'Month','Day', 'Region', 'Summary', 'Target_type', 'Weapon_type', 'Motive'], inplace = True, axis = 1)
	# reorder columns
	cols = ts.columns.tolist()
	cols = cols[-1:] + cols[:-1]
	ts = ts[cols]

	return ts

def show_subset(ts, group):
	"""
	Shows a subsample of time series for a given group.
	"""
	ts = ts[ts['Group'] == group]
	plt.figure(figsize=(15, 12))
	ax1 = plt.subplot(4, 1, 1)
	ax1.set_xticklabels([])
	ax1.set_title('2008')
	ts[(ts['DateTime'] >= '2008-01-01') & (ts['DateTime'] < '2009-01-01')]['DateTime'].value_counts().plot(kind='line', color='#58ACFA', ax=ax1)
	ax2 = plt.subplot(4, 1, 2)
	ax2.set_title('2009')
	ax2.set_xticklabels([])
	ts[(ts['DateTime'] >= '2009-01-01') & (ts['DateTime'] < '2010-01-01')]['DateTime'].value_counts().plot(kind='line', color='#58ACFA', ax=ax2)
	ax3 = plt.subplot(4, 1, 3)
	ax3.set_xticklabels([])
	ax3.set_title('2010')
	ts[(ts['DateTime'] >= '2010-01-01') & (ts['DateTime'] < '2011-01-01')]['DateTime'].value_counts().plot(kind='line', color='#58ACFA', ax=ax3)
	ax4 = plt.subplot(4, 1, 4)
	ax4.set_xticklabels(['February', 'March', 'April', 'May', 'Jun', 'July', 'August', 'September', 'October', 'November', 'December'])
	ax4.set_title('2011')
	ts[(ts['DateTime'] >= '2011-01-01') & (ts['DateTime'] < '2012-01-01')]['DateTime'].value_counts().plot(kind='line', color='#58ACFA', ax=ax4)
	plt.show()