#! /usr/bin/env python3

from tqdm import tqdm
import json

import numpy as np
import pandas as pd

import scipy.sparse.linalg
from scipy import sparse, stats, spatial

import networkx as nx
from networkx.algorithms import community

import matplotlib.pyplot as plt
import seaborn as sns

import random


from helpers_satcat import *

display = print

with open("Dataset/satcat_info.json") as f:
    satcat_info_json = json.load(f)



# Extract the main dataset and put it in a dataframe
sat_info_array = satcat_info_json["sat_data"]
satcat_df = pd.DataFrame(sat_info_array)
satcat_df = satcat_df.set_index("NORAD")
display(satcat_df.head(5))





# Extract complementary information dictionary
operational_status_dict = satcat_info_json["operational_status"]
launch_site_full_name_dict = satcat_info_json["launch_site"]
source_full_name_dict = satcat_info_json["source"]


# #### Fill NaN in satcat data




satcat_df = satcat_df.fillna(value=0)
display(satcat_df.head(5))





num_launch_site_dict = {}
for index,site in enumerate(satcat_df.launch_site.unique()):
    num_launch_site_dict[site] = index

num_launch_site = satcat_df.launch_site.map(lambda x:num_launch_site_dict[x])
num_launch_site.name = "num_launch_site"
satcat_df = pd.concat([satcat_df,num_launch_site], axis=1)

num_operational_status_dict = {}
for index, status in enumerate(satcat_df.operational_status.unique()):
    num_operational_status_dict[status] = index

num_source_dict = {}
for index,site in enumerate(satcat_df.source.unique()):
    num_source_dict[site] = index

orbital_statuses = satcat_df.orbital_status.unique()
num_orbital_status_dict = {}
for index, status in enumerate(orbital_statuses):
    num_orbital_status_dict[status] = index





# Note: because this function is key to understanding how we create the features, we kept it in the notebook

def get_feature_dataframe(reduced_satcat_df, only_payload=True, only_operational=False):
    """Function to create the feature dataframe"""
    # We keep all the features that could have an impact on the clustering
    # Note: We do this in a general fashion, so we keep features that could have been reduced just in case

    # Numeric Features : "apogee", "inclination", "launch_year", "orbital_period", "perigee", "radar_cross_section"
    # Boolean Features : "payload_flag"
    # List Features    : "operational_status", "orbital_status", "source"
    numeric_features = ["apogee", "inclination", "launch_year", "orbital_period", "perigee", "radar_cross_section"]
    boolean_features = ["payload_flag"]
    list_features = ["operational_status", "orbital_status", "source"]
    features_columns = numeric_features + boolean_features + list_features

    # Numeric features don't require special management
    features_df = reduced_satcat_df[numeric_features]
    display(features_df.head(5))

    # Transform boolean features to numeric
    num_payload_flag = reduced_satcat_df.payload_flag.map (                                lambda x : 1 if x else 0
                       )
    if not only_payload:
        features_df = features_df.assign(payload_flag = num_payload_flag)

    # We need to transform the List features in a numerical form, we will use the unique value index to do so
    # We previously created indexes to be able to find them easily
    # "operational_status" : num_operational_status_dict
    # "orbital_status" : num_orbital_status_dict
    # "source": num_source_dict
    num_operational_status = reduced_satcat_df.operational_status.map(                                     lambda x : num_operational_status_dict[x]                              )
    num_orbital_status = reduced_satcat_df.orbital_status.map(                                     lambda x : num_orbital_status_dict[x]                              )
    num_source = reduced_satcat_df.source.map(                                     lambda x : num_source_dict[x]                              )
    if not only_operational:
        features_df = features_df.assign(operational_status = num_operational_status)
    features_df = features_df.assign(orbital_status = num_orbital_status)
    features_df = features_df.assign(source = num_source)

    return features_df


# ### Do analysis with different satellites
#

# We programmed the project for any number of launch sites, but tested it with only two.
#
# What would be the results with a more complex sites of satellites?
#
# Lets try to find out by using the helper function we defined all along the project with different parameters.

# #### Use previous parameters




# REDUCE DATAFRAME PARAMETERS
REDUCE_PER_PERCENTILE = False # Reduction per percentile didn't work as expected
REDUCE_PER_LAUNCH_SITE = True # Attempt to segment by country: there might be reasons for using particular sites
TARGET_PERCENTILE_LAUNCH_SITES = 90
TARGET_PERCENTILE_SOURCES = 90
LAUNCH_SITES = ["AFETR", "AFWTR"]
ONLY_PAYLOAD=True
ONLY_OPERATIONAL=False
SIZE_OF_SMALLEST_CLIQUE = 20

result_dict_normal_param = calculate_all_values(satcat_df,
                         get_feature_dataframe,
                         REDUCE_PER_PERCENTILE,
                         REDUCE_PER_LAUNCH_SITE,
                         TARGET_PERCENTILE_LAUNCH_SITES,
                         TARGET_PERCENTILE_SOURCES,
                         LAUNCH_SITES,
                         ONLY_PAYLOAD,
                         ONLY_OPERATIONAL,
                         SIZE_OF_SMALLEST_CLIQUE
                        )
error_normal_param = calculate_error(result_dict_normal_param["reduced_satcat_df"],
                                     result_dict_normal_param["subgraphs"])
print_error_graph(error_normal_param, "error_normal_param")


# #### Use percentile of launch sites




REDUCE_PER_PERCENTILE = True
REDUCE_PER_LAUNCH_SITE = False
TARGET_PERCENTILE_LAUNCH_SITES = 80
TARGET_PERCENTILE_SOURCES = 0

result_dict_launch_sites_perc = calculate_all_values(satcat_df,
                         get_feature_dataframe,
                         REDUCE_PER_PERCENTILE,
                         REDUCE_PER_LAUNCH_SITE,
                         TARGET_PERCENTILE_LAUNCH_SITES,
                         TARGET_PERCENTILE_SOURCES,
                         LAUNCH_SITES,
                         ONLY_PAYLOAD,
                         ONLY_OPERATIONAL,
                         SIZE_OF_SMALLEST_CLIQUE
                        )
error_launch_sites_perc = calculate_error(result_dict_launch_sites_perc["reduced_satcat_df"],
                                     result_dict_launch_sites_perc["subgraphs"])

print_error_graph(error_launch_sites_perc, "error_launch_sites_perc")
plt.close()


# #### Use more launch sites




# Use top 10 sites with the most launches
launches_per_site.head(10)





REDUCE_PER_PERCENTILE = False
REDUCE_PER_LAUNCH_SITE = True
LAUNCH_SITES = ["AFETR", "AFWTR", "TYMSC", "PLMSC", "TAISC", "FRGUI", "SRILR", "XICLF", "JSC", "KYMSC"]

result_dict_many_launch_sites = calculate_all_values(satcat_df,
                         get_feature_dataframe,
                         REDUCE_PER_PERCENTILE,
                         REDUCE_PER_LAUNCH_SITE,
                         TARGET_PERCENTILE_LAUNCH_SITES,
                         TARGET_PERCENTILE_SOURCES,
                         LAUNCH_SITES,
                         ONLY_PAYLOAD,
                         ONLY_OPERATIONAL,
                         SIZE_OF_SMALLEST_CLIQUE
                        )
error_many_launch_sites = calculate_error(result_dict_many_launch_sites["reduced_satcat_df"],
                                     result_dict_many_launch_sites["subgraphs"])

print_error_graph(error_many_launch_sites, "error_many_launch_sites")
plt.close()


# #### Use all satellites




# REDUCE DATAFRAME PARAMETERS
REDUCE_PER_PERCENTILE = True # Reduction per percentile didn't work as expected
REDUCE_PER_LAUNCH_SITE = False # Attempt to segment by country: there might be reasons for using particular sites
TARGET_PERCENTILE_LAUNCH_SITES = 0
TARGET_PERCENTILE_SOURCES = 0
LAUNCH_SITES = ["AFETR", "AFWTR"]
ONLY_PAYLOAD=True
ONLY_OPERATIONAL=False
SIZE_OF_SMALLEST_CLIQUE = 20

result_dict_all_sats = calculate_all_values(satcat_df,
                         get_feature_dataframe,
                         REDUCE_PER_PERCENTILE,
                         REDUCE_PER_LAUNCH_SITE,
                         TARGET_PERCENTILE_LAUNCH_SITES,
                         TARGET_PERCENTILE_SOURCES,
                         LAUNCH_SITES,
                         ONLY_PAYLOAD,
                         ONLY_OPERATIONAL,
                         SIZE_OF_SMALLEST_CLIQUE
                        )
error_all_sats = calculate_error(result_dict_all_sats["reduced_satcat_df"],
                                     result_dict_all_sats["subgraphs"])
print_error_graph(error_all_sats, "error_all_sats")
plt.close()

