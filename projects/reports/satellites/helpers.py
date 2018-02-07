import numpy as np
from numpy import pi
import pandas as pd
import ephem
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from datetime import datetime, timedelta



def read_tle(tle_file):
    """
    Helper function that read a convert a tle file to an array
    :param tle_file: The tle file to read
    :return: the tle file as an array
    """
    tle = open(tle_file)
    lines = tle.readlines()
    return lines

def compute_tle_informations(lines):
    """
    Compute the positions (latitudes, longitudes, elevations) of each satellites
    :param lines: the tle file as an array
    :return: the names of each satellites
    :return: the longitude of each satellites
    :return: the latitude of each satellites
    :return: the elevation of each satellites
    """
    numb_element = len(lines)
    names = []
    long = []
    lat = []
    elevation_km = []
    for i in range(0,numb_element-1,3):
        temp = ephem.readtle(lines[i], lines[i+1], lines[i+2])
        temp.compute('2017/10/12')
        try:
            long.append(temp.sublong*180/pi)
            lat.append(temp.sublat*180/pi)
            elevation_km.append(temp.elevation/1000)
            names.append(temp.name[2:])
        except:
            print('Index number {} is not compatible to perform computation' .format(i))
    return names,long,lat,elevation_km

def satellite_orbit(name, sioi, dict_tle):
    """
    Compute the orbit of each satellites by sampling 200 positions of a satellites during one period
    :param name: the names of the satellites
    :param sioi: the "satellites_in_orbit_info" file used to get the period of a satellite
    :param dict_tle: the tle file as a dict
    :return: an 3 dimensional array representing the orbit of each satellites. The array is
    of dimension [nb_satellites X 200 X 2], where 200 is the sampling number and 2 is the
    position (latitude, longitude)
    """
    orbit_point = []
    nb_point = 200
    period = int(sioi[sioi.OBJECT_NAME == name].PERIOD.values[0])
    delta_t = int(period*60/nb_point)
    delta_t = timedelta(seconds = delta_t)
    time = datetime(2017,10,12)
    l1, l2 = dict_tle[name]
    temp = ephem.readtle(name, l1, l2)
    for i in range(0,200):
        time = time + delta_t
        temp.compute(time)
        try:
            lat = temp.sublat*180/pi
            long = temp.sublong*180/pi
            orbit_point.append((lat,long))
        except:
            print("ERROR")
    return orbit_point


def compute_grids(res):
    """
    Compute a "grid" that represent the surface of the earth which was overflown by each satellites
    :param res: the orbit representation of each satellites returned by the satellites_orbit function
    :return: the grid representation of the orbit of each satellites
    """
    orbit_grids= []
    vertical_offset =90
    horizontal_offset = 180
    shape = (180,360)
    coverage_width = range(-7, 7)
    for stats in res:
        grid = np.zeros(shape)
        for lat, long in stats:
            #add the offset for each component to have 0<lat<180 and 0<long<360
            lat = round(lat) + vertical_offset
            long = round(long) + horizontal_offset
            #fill the neighborhood of the projection to have some continuous representation
            for i in coverage_width:
                for j in coverage_width:
                    grid[(lat+i)%shape[0]][(long+j)%shape[1]] = 1
        orbit_grids.append(grid)
    return orbit_grids


def plot_map(data_final,labels1=None,label_value=None):
    """
    Plot the world map with the sattelites positions as yellow dots
    :param data_final: the list of satellites with there positions at a given time (2017-10-12 in our case)
    :param labels1: the cluster labels  of each satellites
    :param label_value: the cluster label to be plotted
    :return: plot the world map with yellow point representing the satellites positions
    """
    plt.figure(figsize=(20,10))
    eq_map = Basemap(projection='robin', resolution = 'l', area_thresh = 1000.0,
                  lat_0=0,lon_0=0)
    eq_map.drawcoastlines()
    eq_map.drawcountries()
    eq_map.bluemarble()
    eq_map.drawmapboundary()
    eq_map.drawmeridians(np.arange(0, 360, 30))
    eq_map.drawparallels(np.arange(-90, 90, 30))

    if labels1 is not None and label_value is not None:
        long = data_final['Longitude [°]'].values[labels1==label_value]
        lat = data_final['Latitude [°]'].values[labels1==label_value]
    else:
        long = data_final['Longitude [°]'].values
        lat = data_final['Latitude [°]'].values
    lons,lats = eq_map(long,lat)
    eq_map.scatter(lons, lats, marker = 'o', color='y', zorder=1)
    plt.show()

def dbscan_func(G, epsilon, mn, labels=None, label_value=None):
    """
    Run a db scan algorithm that performs simple unsupervised clustering assignement
    :param G: The weighted graph where the weight is the common overflown surface
    :param epsilon: Size of the neighborhood for one point
    :param mn: Minimum number of datapoints within the neighborhood
    :param labels: If not none, precomputed label from a previous clustering
    :param label_value: If not none, a specific label from a previous clustering
    :return: plot the clustering assignement and return the new labels.
    """
    if labels is not None and label_value is not None:
        X = G.coords[labels==label_value]
    else:
        X = G.coords

    X = StandardScaler().fit_transform(X)
    db = DBSCAN(eps= epsilon, min_samples = mn).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels1 = db.labels_

    n_clusters_ = len(set(labels1)) - (1 if -1 in labels1 else 0)
    unique_labels = set(labels1)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels1 == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
    return labels1

def compute_weight(sat1,sat2):
    """
    Compute the weight of the edge between two satellites. The weight is defined as the common surface overflown
    by both satellites
    :param sat1: the first satellite
    :param sat2: the second satellite
    :return: the weight of the edge between the two satellites
    """
    tot = np.logical_and(sat1,sat2)
    return np.sum(tot)


def compute_adja(orbit_grids):
    """
    Compute the adjacency matrix
    :param orbit_grids: the grid representation of the orbits for each satellites
    :return: The adjacency matrix
    """
    adja = np.zeros([len(orbit_grids),len(orbit_grids)])
    for n1,i1 in enumerate(orbit_grids):
        for n2,i2 in enumerate(orbit_grids[n1+1:]):
            adja[n1,n2+n1+1] = compute_weight(i1,i2)
    return adja + np.transpose(adja)
