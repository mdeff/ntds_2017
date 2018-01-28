"""
Helper functions
"""

from graph_tool.all import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from pygsp import graphs, filters, plotting
import configparser
import requests
from mpl_toolkits.basemap import Basemap
from tqdm import tqdm

credentials = configparser.ConfigParser()
credentials.read('credentials.ini')

INTRINIO_USERNAME = credentials.get('intrinio', 'api_username')
INTRINIO_PASSWORD = credentials.get('intrinio', 'api_password')

GOOGLE_MAPS_API_URL = 'https://maps.googleapis.com/maps/api/geocode/json'
GOOGLE_MAPS_API_KEY = credentials.get('google_maps', 'api_key')

def build_bipartite_investments_graph(investments, funding_rounds):
    g = Graph()
    vertices = {} #Dictionary keeping in memory the mapping between ids and vertex objects.

    # Vertex properties for g
    g.vp.vertex_id = g.new_vertex_property("string") #The id of the company

    # Edge properties for g
    g.ep.funding_round = g.new_edge_property("float") #Amount of the funding round

    for _, investment in investments.iterrows():
        investor_id = investment["investor_object_id"]
        funded_id = investment["funded_object_id"]
        funding_round_id = investment["funding_round_id"]

        if investor_id not in vertices:
            v1 = g.add_vertex()
            vertices[investor_id] = v1
            g.vp.vertex_id[v1] = investor_id

        if funded_id not in vertices:
            v2 = g.add_vertex()
            vertices[funded_id] = v2
            g.vp.vertex_id[v2] = funded_id

        e = g.add_edge(vertices[investor_id], vertices[funded_id])
        g.ep.funding_round[e] = funding_rounds.loc[funding_round_id, "raised_amount_usd"]

    return g, vertices

def build_investments_graph(investments, companies, funding_rounds):
    g1, vertices = build_bipartite_investments_graph(investments, funding_rounds)

    g2 = Graph(directed=False)
    funded_nodes = {}

    # Vertex properties for g2
    g2.vp.vertex_id = g2.new_vertex_property("string") #The id of the company
    g2.vp.category_code = g2.new_vertex_property("int") #The category code of the company
    g2.vp.category_name = g2.new_vertex_property("string") #The category name of the company
    g2.vp.funding_total_usd = g2.new_vertex_property("float") #The total amount raised by the company
    g2.vp.lat = g2.new_vertex_property("float") #The latitude of the company
    g2.vp.lng = g2.new_vertex_property("float") #The longitude of the company
    g2.vp.ROI = g2.new_vertex_property("float") #Ratio Price Acquisition/Funding Total USD

    # List of investors
    investors_id = np.unique(investments["investor_object_id"])

    for investor_id in investors_id:
        investor_node = vertices[investor_id]
        neighbors = np.unique(g1.get_out_neighbors(investor_node))
        nbr_neighbors = len(neighbors)

        #Looping for each pair of companies
        for i in range(nbr_neighbors - 1):
            for j in np.arange(i + 1, nbr_neighbors):

                id_funded_node1 = g1.vp.vertex_id[g1.vertex(neighbors[i])]
                id_funded_node2 = g1.vp.vertex_id[g1.vertex(neighbors[j])]

                if id_funded_node1 not in funded_nodes:
                    funded_node1 = g2.add_vertex()
                    g2.vp.vertex_id[funded_node1] = id_funded_node1
                    category_code, category_name = get_parent_category(companies.loc[id_funded_node1, "category_code"])
                    g2.vp.category_name[funded_node1] = category_name
                    g2.vp.category_code[funded_node1] = category_code
                    g2.vp.funding_total_usd[funded_node1] = companies.loc[id_funded_node1, "funding_total_usd"]
                    g2.vp.lat[funded_node1] = companies.loc[id_funded_node1, "lat"]
                    g2.vp.lng[funded_node1] = companies.loc[id_funded_node1, "lng"]
                    g2.vp.ROI[funded_node1] = companies.loc[id_funded_node1, "ROI"]
                    funded_nodes[id_funded_node1] = funded_node1

                if id_funded_node2 not in funded_nodes:
                    funded_node2 = g2.add_vertex()
                    g2.vp.vertex_id[funded_node2] = id_funded_node2
                    category_code, category_name = get_parent_category(companies.loc[id_funded_node2, "category_code"])
                    g2.vp.category_name[funded_node2] = category_name
                    g2.vp.category_code[funded_node2] = category_code
                    g2.vp.funding_total_usd[funded_node2] = companies.loc[id_funded_node2, "funding_total_usd"]
                    g2.vp.lat[funded_node2] = companies.loc[id_funded_node2, "lat"]
                    g2.vp.lng[funded_node2] = companies.loc[id_funded_node2, "lng"]
                    g2.vp.ROI[funded_node2] = companies.loc[id_funded_node2, "ROI"]
                    funded_nodes[id_funded_node2] = funded_node2

                g2.add_edge(funded_nodes[id_funded_node1], funded_nodes[id_funded_node2])

    return g2



def get_parent_category(category):
    if category in ["software", "web", "mobile", "network_hosting", "search", "social", "messaging", "analytics", "ecommerce", "games_video"]:
        return 1, "information_technology"
    elif category in ["advertising", "consulting", "public_relations", "legal", "finance"]:
        return 2, "consulting_services"
    elif category in ["hardware", "semiconductor", "manufacturing", "transportation", "automotive", "cleantech"]:
        return 3, "industry"
    elif category in ["health", "medical", "biotech", "nanotech"]:
        return 4, "health"
    elif category in ["sports", "music", "photo_video", "pets", "news", "design", "fashion", "travel", "hospitality"]:
        return 5, "entertainment"
    elif category in ["other", "enterprise", "local", "nonprofit", "government", "security", "education", "real_estate"]:
        return 6, "other"
    else:
        return 6, "other"



def plot_country(companies):
    countries=companies['country_code'].value_counts()
    sumOthers = np.array(list(countries.values))[9:].sum()
    countries_p = list(np.array(list(countries.values))[0:9]) + [sumOthers]
    names_countries =list(countries[:9].index)+ ['Other']
    patches,_ = plt.pie(countries_p)
    plt.legend(patches, names_countries, loc="best")
    plt.axis('equal')
    
def plot_categories(companies):
    category_counts = companies["category_code"].value_counts()
    labels = category_counts.index.tolist()
    sizes = category_counts.tolist()
    patches,_ = plt.pie(sizes)
    plt.legend(patches, labels, loc="best")
    plt.axis('equal')

def merge_categories(categories):
    parent_categories = {
        "information_technology": ["software", "web", "mobile", "network_hosting", "search", "social", "messaging", "analytics", "ecommerce", "games_video"],
        "consulting_services": ["advertising", "consulting", "public_relations", "legal", "finance"],
        "industry": ["hardware", "semiconductor", "manufacturing", "transportation", "automotive", "cleantech"],
        "health": ["health", "medical", "biotech", "nanotech"],
        "entertainment": ["sports", "music", "photo_video", "pets", "news", "design", "fashion", "travel", "hospitality"],
        "other": ["other", "enterprise", "local", "nonprofit", "government", "security", "education", "real_estate"]
    }
    mapping = {v: k for k,vv in parent_categories.items() for v in vv}

    return categories.map(mapping)


def get_market_cap(symbol):
    """
    Given a company symbol on the stock market, returns its market capitalization (of end 2013) thanks to the Intrionio API.
    """
    base_url = "https://api.intrinio.com/historical_data"
    symbol = symbol.split(":")[-1]
    response = requests.get("{}?identifier={}&item=marketcap&start_date=2013-12-31&end_date=2014-01-01".format(base_url, symbol), auth=(INTRINIO_USERNAME, INTRINIO_PASSWORD))
    
    try:
        return response.json()["data"][0]["value"]
    except:
        return float("NaN")

def compute_companies(G, ROI, start_indices, tau, nb_best_companies):
    """
    Algorithm that computes neighbors of starting vertices by filtering a signal according to the Heat diffusion equation.
    """
    f = filters.Heat(G, tau)
    
    weighted_delta = np.zeros(G.N)
    weighted_delta[start_indices] = ROI[start_indices] / np.sum(ROI[start_indices])
    
    s = f.filter(weighted_delta)
    
    plt.subplot(211)
    plt.plot(weighted_delta)
    plt.subplot(212)
    plt.plot(s)
    
    company_indices = np.flip(np.argsort(s),0)[:nb_best_companies]
    threshold = s[company_indices[-1]]
    plt.plot([threshold]*G.N, "r")
    
    return list(set(company_indices) - set(start_indices))

def compute_return_on_investment(company_indices, companies, vertex_ids, ipos, risk):
    
    multipliers = []
    stats = {
        "operating": 0,
        "acquired": 0,
        "closed": 0,
        "ipo": 0
    }
    
    
    for company_index in tqdm(company_indices):
        vertex_index = vertex_ids[company_index].decode("utf-8") 
        company = companies.loc[vertex_index]
        
        if company["status"] == "acquired" and not np.isnan(company["ROI"]):
            multiplier = company["ROI"]
        elif company["status"] == "closed":
            multiplier = 0
        elif company["status"] == "ipo":
            valuation_amount = ipos.loc[vertex_index, "valuation_amount"]
            funding_total = company["funding_total_usd"]
            if not np.isnan(valuation_amount) and not np.isnan(funding_total):
                multiplier = valuation_amount / funding_total
            elif not np.isnan(funding_total):
                symbol = ipos.loc[vertex_index, "stock_symbol"]
                market_cap = get_market_cap(symbol)
                if not np.isnan(market_cap):
                    multiplier = market_cap / funding_total
        else:
            #We don't have any information about the price it has been or will be sold
            multiplier = risk
        
        multipliers.append(multiplier)
        stats[company["status"]] += 1
        
    return multipliers, stats

def draw_map(lat, lng, title, size=2.5, alpha=1.):
    """
    Plots points given by their latitude and longitude on a map.
    """
    fig=plt.figure(figsize=(18,24))
    m = Basemap(resolution='c')
    x, y = m(lng, lat)
    m.scatter(x, y, marker=".", c='darkred', s=size, alpha=alpha, zorder=3)
    #m.drawcoastlines()
    m.drawmapboundary(fill_color='lightblue')
    m.fillcontinents(color='wheat')
    plt.title(title)
    plt.show()

def get_lat_lng(country_code, city):
    """
    Given a country code and a city, returns the latitude and longitude of the location thanks to the Google Maps API.
    """
    params = {
        "address": city,
        "country": country_code,
        'sensor': 'false',
        "key": GOOGLE_MAPS_API_KEY
    }
    req = requests.get(GOOGLE_MAPS_API_URL, params=params)
    res = req.json()

    if len(res['results']) > 0:
        result = res['results'][0]
        lat = result['geometry']['location']['lat']
        lng = result['geometry']['location']['lng']
    else:
        lat = float("NaN")
        lng = float("NaN")

    return lat, lng

def geocode_companies(companies):
    """
    Given the companies database, computes the latitude and longitude of each company and memorizes the already seen locations to avoid using the API too much.
    """
    
    def isNaN(x):
        return x != x
    
    def is_new_location(country_code, city):
        return len(locations[(locations["country_code"] == country_code) & (locations["city"] == city)]) == 0


    def get_location(country_code, city):
        return locations[(locations["country_code"] == country_code) & (locations["city"] == city)].iloc[0]


    def add_new_location(country_code, city, lat, lng):
        return locations.append({"country_code": country_code, "city": city, "lat": lat, "lng": lng}, ignore_index=True)
    
    locations = pd.DataFrame({ "city": [""], "country_code": [""], "lat": [0], "lng": [0] })
    
    N = len(companies)

    for i in tqdm(range(N)):
        company = companies.iloc[i]

        country_code, city = company["country_code"], company["city"]

        if isNaN(city) or isNaN(country_code):
            continue

        if is_new_location(country_code, city):
            lat, lng = get_lat_lng(country_code, city)
            locations = add_new_location(country_code, city, lat, lng)
        else:
            location = get_location(country_code, city)
            lat, lng = location["lat"], location["lng"]

        companies.loc[company.name, "lat"] = lat
        companies.loc[company.name, "lng"] = lng
    
    return companies, locations

def compute_ROI(companies, acquisitions):
    N = len(acquisitions)

    for i in tqdm(range(N)):
        acquisition = acquisitions.iloc[i]

        try:
            funding_total_usd = companies.loc[acquisition.name, "funding_total_usd"]
        except:
            continue

        acquisition_amount = acquisition["price_amount"]

        if acquisition["price_currency_code"] == "USD" and funding_total_usd > 0 and acquisition_amount > 0:
            companies.loc[acquisition.name, "ROI"] = acquisition_amount / funding_total_usd

def trim_investments(investments, companies, threshold):
    indices_to_remove = []

    for i, investment in investments.iterrows():
        try:
            company = companies.loc[investment["funded_object_id"]]

            if np.isnan(company["lat"]) or np.isnan(company["lng"]):
                indices_to_remove.append(i)

            if np.isnan(company["funding_total_usd"]) or company["funding_total_usd"] < threshold:
                indices_to_remove.append(i)
        
        except:
            indices_to_remove.append(i)
    
    investments = investments.drop(indices_to_remove)
    
    return investments