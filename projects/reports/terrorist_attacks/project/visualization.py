"""
Visualization module.
"""

import numpy as np
from matplotlib import animation
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

from pca import create_handles


import warnings
warnings.filterwarnings('ignore')

def get_temp_markers(year, attacks):
  """
  Gives all the information about the markers needed for the 
  year passed in argument.
  """
  data_given_year = attacks[attacks['Year'] == year].reset_index()
  num_markers = data_given_year.shape[0]
  markers = np.zeros(num_markers, dtype=[('Longitude', float, 1),
                                    ('Latitude', float, 1),
                                    ('Size',     float, 1),
                                    ('Color',    float, 1)])
  
  killed = data_given_year['Killed']
  _MIN, _MAX, _MEDIAN = killed.min(), killed.max(), killed.median()

  markers['Longitude'] = data_given_year['Longitude']
  markers['Latitude'] = data_given_year['Latitude']
  markers['Size'] = 10* np.abs(killed - _MEDIAN) + 1
  markers['Color'] = (killed - _MIN)/(_MAX - _MIN)
  
  return markers, _MAX

def world_view(attacks):
  """
  Creates an animation where we see the evolution of the worldwide terrorist attacks
  among the available years.
  """

  fig = plt.figure(figsize=(10, 10))
  cmap = plt.get_cmap('inferno')

  # create the map
  map = Basemap(projection='cyl')
  map.drawmapboundary()
  map.fillcontinents(color='lightgray', zorder=0)

  # define the frame values (as 1993 is not contained in the database
  # we have to remove it, otherwise we will have an empty frame)
  frames = np.append(np.arange(1970, 1993), np.arange(1994, 2017))

  # create the plot structure
  temp_markers, _MAX = get_temp_markers(frames[0], attacks)
  xs, ys = map(temp_markers['Longitude'], temp_markers['Latitude'])
  scat = map.scatter(xs, ys, s=temp_markers['Size'], c=temp_markers['Color'], cmap=cmap, marker='o', 
    alpha=0.3, zorder=10)

  year_text = plt.text(-170, 80, str(frames[0]),fontsize=15)
  cbar = map.colorbar(scat, location='bottom')
  cbar.set_label('number of killed people 0.0 = min [0] 1.0 = max [{}]' .format(_MAX))
  plt.title('Activity of terrorism attacks from 1970 to 2016')
  plt.savefig('world_view.pdf', bbox_inches='tight')
  plt.show()

  def update(year):
    """
    Updates the content of each frame during the animation for
    the year passed in argument. 
    """
    # retrieve necessary information from the markers
    temp_markers, _MAX = get_temp_markers(year, attacks)

    # update the map content
    xs, ys = map(temp_markers['Longitude'], temp_markers['Latitude'])
    scat.set_offsets(np.hstack((xs[:,np.newaxis], ys[:, np.newaxis])))
    scat.set_color(cmap(temp_markers['Color']))
    scat.set_sizes(temp_markers['Size'])
    year_text.set_text(str(year))
    cbar.set_label('number of killed people 0.0 = min [0] 1.0 = max [{}]' .format(_MAX))

    return scat,

  # create animation
  ani = animation.FuncAnimation(fig, update, interval=1000, frames=frames, blit=True)
  ani.save('visualization.mp4', writer = 'ffmpeg', fps=1, bitrate=-1)

  plt.show()

def get_group_markers(attacks, group):
  """
  Gives all the information about the markers for the 
  group passed in argument.
  """

  data_given_group = attacks[attacks['Group'] == group]
  num_markers = data_given_group.shape[0]
  markers = np.zeros(num_markers, dtype=[('Longitude', float, 1),
                                ('Latitude', float, 1),
                                ('Size',     float, 1),
                                ('Color',    float, 1)])

  killed = data_given_group['Killed']
  _MIN, _MAX, _MEDIAN = killed.min(), killed.max(), killed.median()

  markers['Longitude'] = data_given_group['Longitude']
  markers['Latitude'] = data_given_group['Latitude']
  markers['Size'] = 10* np.abs(killed - _MEDIAN) + 1
  markers['Color'] = (killed - _MIN)/(_MAX - _MIN)

  return markers, _MAX

def zoom_taliban_intensity(attacks):
  """
  Zooms in the particular location of the attacks perpetrated by the Taliban group
  showing the intensity of the attacks.
  """

  fig = plt.figure(figsize=(15,15))
  ax = fig.add_subplot(111)
  cmap = plt.get_cmap('inferno')

  plt.title('Intensity of attacks perpetrated by the Taliban group\n')

  # create the map
  map = Basemap(projection='cyl',lat_0=0, lon_0=0)
  map.drawmapboundary()
  map.fillcontinents(color='lightgray', zorder=0)

  # create the plot structure
  temp_markers, _MAX = get_group_markers(attacks, 'Taliban')
  xs, ys = map(temp_markers['Longitude'], temp_markers['Latitude'])
  scat = map.scatter(xs, ys, s=temp_markers['Size'], c=temp_markers['Color'], cmap=cmap, marker='o', 
  alpha=0.3, zorder=10)

  axins = zoomed_inset_axes(ax, 9, loc=2)
  axins.set_xlim(25, 40)
  axins.set_ylim(60, 75)

  plt.xticks(visible=False)
  plt.yticks(visible=False)

  map2 = Basemap(llcrnrlon=55,llcrnrlat=25,urcrnrlon=75,urcrnrlat=40, ax=axins)
  map2.drawmapboundary()
  map2.fillcontinents(color='lightgray', zorder=0)
  map2.drawcoastlines()
  map2.drawcountries()

  map2.scatter(xs, ys, s=temp_markers['Size']/5., c=cmap(temp_markers['Color']), alpha=0.5)
  mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

  plt.savefig('taliban_zoom_intensity.pdf', bbox_inches='tight')
  plt.show()

def get_group_attack_types_markers(attacks, group):
  """
  Gives the description of the attack types about the markers for the 
  group passed in argument.
  """
  data_given_year = attacks[attacks['Group'] == group]
  list_attack_type_unique = data_given_year['Attack_type'].unique().tolist()
  list_attack_type = data_given_year['Attack_type'].tolist()
  
  
  # assign each attack to the corresponding color
  colors_attack_type = plt.cm.tab20(list(range(1,len(list_attack_type_unique)+1)))
  label_color_dict_attack_type = dict(zip(list_attack_type_unique, colors_attack_type))
  cvec_attack_type = [label_color_dict_attack_type[label] for label in list_attack_type]

  num_markers = data_given_year.shape[0]
  markers = np.zeros(num_markers, dtype=[('Longitude', float, 1),
                                  ('Latitude', float, 1),
                                  ('Size',     float, 1),
                                  ('Color',    float, 4)])

  killed = data_given_year['Killed']
  _MIN, _MAX, _MEDIAN = killed.min(), killed.max(), killed.median()

  markers['Longitude'] = data_given_year['Longitude']
  markers['Latitude'] = data_given_year['Latitude']
  markers['Size'] = 100
  markers['Color'] = np.array(cvec_attack_type)

  return markers, label_color_dict_attack_type

def zoom_taliban_attack_types(attacks):
  """
  Zooms in the particular location of the attacks perpetrated by the Taliban group
  showing the different attack types.
  """

  group = 'Taliban'
  fig = plt.figure(figsize=(15,15))
  ax = fig.add_subplot(111)
  cmap = plt.get_cmap('inferno')

  plt.title('Attack types perpetrated by the Taliban group\n')

  # create the map
  map = Basemap(projection='cyl',lat_0=0, lon_0=0)
  map.drawmapboundary()
  map.fillcontinents(color='lightgray', zorder=0)

  # create the plot structure
  temp_markers, _MAX = get_group_markers(attacks, group)
  xs, ys = map(temp_markers['Longitude'], temp_markers['Latitude'])
  scat = map.scatter(xs, ys, s=temp_markers['Size'], c=temp_markers['Color'], cmap=cmap, marker='o', 
  alpha=0.5, zorder=10)

  axins = zoomed_inset_axes(ax, 9, loc=2)
  axins.set_xlim(25, 40)
  axins.set_ylim(60, 75)

  plt.xticks(visible=False)
  plt.yticks(visible=False)

  map2 = Basemap(llcrnrlon=55,llcrnrlat=25,urcrnrlon=75,urcrnrlat=40, ax=axins)
  map2.drawmapboundary()
  map2.fillcontinents(color='lightgray', zorder=0)
  map2.drawcoastlines()
  map2.drawcountries()


  temp_markers, label_color_dict_attack_type = get_group_attack_types_markers(attacks, group)
  map2.scatter(xs, ys, s=temp_markers['Size']/5., c=temp_markers['Color'], alpha=0.5)

  mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

  handles = create_handles(label_color_dict_attack_type, ax)
  labels = [h.get_label() for h in handles] 
  ax.legend(loc='upper left', bbox_to_anchor=(1, 1), handles=handles, labels=labels)  

  plt.savefig('taliban_zoom_attack_types.pdf', bbox_inches='tight')
  plt.show()

