import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.neighbors import kneighbors_graph
import networkx as nx
import numpy as np
from sklearn import preprocessing

# Function to do some comparison between professionals and students
def plot_stud_prof(prof_stack="", stud_stack="", prof="", stud="", column="", title=""):
    if column!="":
        prof=prof_stack[column]
        stud=stud_stack[column]
    p = prof.value_counts(normalize=True)[:10]
    v_stud = stud.value_counts(normalize=True)
    s = v_stud.loc[p.index]
    df = pd.DataFrame([p, s])
    df = df.T
    df.columns = ["Professional", "Student"]
    df.plot.bar(figsize=(7,7))
    plt.title(title + ' distribution for Professionals/Students')
    plt.ylabel("Ratio")
    plt.show()
    
    
# We filter out the devs upon some criteria
def row_filter(stack, row):
    if row.Professional not in ["Student", 
                                "Professional developer"]:
        return False
    if row.Professional == "Professional developer":
        if row.EmploymentStatus not in ['Employed part-time',
                                        'Employed full-time',
                                        'Independent contractor, freelancer, or self-employed']:
            return False
        # After checking salary values, we decided to remove the first 5%
        # quantile as they were mostly outliers (values inbetween 0 and 100)
        if row.isnull().Salary or row.Salary < stack.Salary.quantile(0.05):
            return False
        if row.isnull().JobSatisfaction and row.isnull().CareerSatisfaction:
            return False
    else:
        if row.isnull().ExpectedSalary or row.ExpectedSalary < stack.ExpectedSalary.quantile(0.05):
            return False
    return True

#Dummies the dataframe
def dummies(df, columns, special_col):
        
    for sub in columns:
        df[sub] = df[sub].apply(lambda x: str(x).replace(" ", "").split(";"))
        if sub == special_col:
            df[sub] = df[sub].apply(lambda x: ["Want_" + s for s in x])
        df = pd.concat([df, pd.get_dummies(pd.DataFrame(df[sub].tolist(), index=df.index).stack()).sum(level=0)], axis=1).drop(sub, axis=1)
    df = pd.get_dummies(df)
    return df

#Preprocess dataframe (dummies and nan)
def preprocessed(df, columns, special_col, prof):
    df = df.dropna()
    final_df = dummies(df.copy(), columns, special_col)
    if prof:
        final_df.JobSatisfaction /= 10
        final_df.CareerSatisfaction /= 10
    return final_df, df

#Compute knn graph using sklearn
def compute_knn_graph(df):
    graph = kneighbors_graph(df, int(np.sqrt(df.shape[0])), mode='distance', include_self=True)
    graph.data = np.exp(- graph.data ** 2 / (2. * np.mean(graph.data) ** 2))
    return graph
             
#Draw corresponding graph using networkX
def draw_graph(graph, title):
    G = nx.from_scipy_sparse_matrix(graph,edge_attribute='similarity')
    pos = nx.spring_layout(G)
    plt.figure(1,figsize=(10,10))
    nx.draw_networkx_nodes(G, pos, node_size=7, node_color='lightblue')
    plt.title(title)
    plt.show()
    return G, pos

#Encode string label to int
def encode_label(df, features):

    mapping_prof = []
    df_encode = df.copy()
    for c in features:
        le = preprocessing.LabelEncoder()
        le.fit(df_encode[c])
        df_encode[c] = le.transform(df_encode[c])
        le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    
        mapping_prof.append(le_name_mapping)
        
    return mapping_prof, df_encode

#Plot graph by features
def draw_features(important_features, df, mapping, G, pos, type_):
    for i,features in enumerate(important_features):
        #print(features)
        f = plt.figure(1,figsize=(10,10))
        norm = plt.Normalize()
        cmap = plt.get_cmap('Set2')
        c = cmap(norm(list(df[features])))
        if i in [0,2]:
            scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
            ax = f.add_subplot(1,1,1)
            for label in mapping[i]:
                ax.plot([0],[0],color=scalarMap.to_rgba(mapping[i][label]),label=label)
    
        nx.draw_networkx_nodes(G, pos, node_color=c, node_size=20)
        plt.legend()
        plt.title(features + " coloring for " + type_ + " network")
        plt.show()
        
#Draw the neighbors of a certain node
def draw_neighbors(G, pos, node, title):
    color = ['lightblue'] * len(G.node)
    color[node] = 'k'
    for n in G.neighbors(node):
        if n != node:
            color[n] = 'r'
    plt.figure(1,figsize=(10,10))
    nx.draw_networkx_nodes(G, pos, node_color=color, node_size=20)
    plt.title(title)
    plt.show()

MAP_COUNTRIES = {'Afghanistan':'AFG',
'Aland Islands':'ALA','Albania':'ALB','Algeria':'DZA','American Samoa':'ASM','Andorra':'AND','Angola':'AGO','Anguilla':'AIA','Antigua and Barbuda':'ATG',
'Antarctica':'ATA','Argentina':'ARG','Armenia':'ARM','Aruba':'ABW','Australia':'AUS','Austria':'AUT','Azerbaijan':'AZE','Azerbaidjan':'AZE',
'Bahrain':'BHR','Bahamas':'BHS','Bangladesh':'BGD','Barbados':'BRB','Belarus':'BLR',
'Belgium':'BEL','Belize':'BLZ','Benin':'BEN','Bermuda':'BMU','Bhutan':'BTN','Bolivia':'BOL','Bosnia and Herzegovina':'BIH','Bosnia-Herzegovina':'BIH',
'Botswana':'BWA','Bouvet Island':'BVT','Brazil':'BRA','British Virgin Islands':'VGB','British Indian Ocean Territory':'IOT',
'Brunei':'BRN','Brunei Darussalam':'BRN','Bulgaria':'BGR','Burkina Faso':'BFA','Burma':'MMR',
'Burundi':'BDI','Cabo Verde':'CPV','Cape Verde':'CPV','Cambodia':'KHM','Cameroon':'CMR',
'Canada':'CAN','Cayman Islands':'CYM','Central African Republic':'CAF','Chad':'TCD','Chile':'CHL',
'Christmas Island':'CHR','China':'CHN','Colombia':'COL','Comoros':'COM','Congo, Democratic Republic of the':'COD',
'Congo, Republic of the':'COG','Cook Islands':'COK','Costa Rica':'CRI','Cote d\'Ivoire':'CIV',
"Ivory Coast (Cote D'Ivoire)":'CIV','Croatia':'HRV','Cuba':'CUB','Curacao':'CUW','Cyprus':'CYP',
'Czech Republic':'CZE','Denmark':'DNK','Djibouti':'DJI','Dominica':'DMA','Dominican Republic':'DOM',
'Ecuador':'ECU','Egypt':'EGY','El Salvador':'SLV','Equatorial Guinea':'GNQ','Eritrea':'ERI','Estonia':'EST',
'Ethiopia':'ETH','Falkland Islands (Islas Malvinas)':'FLK','Falkland Islands':'FLK','Faroe Islands':'FRO',
'Fiji':'FJI','Finland':'FIN','France':'FRA','French Polynesia':'PYF','Gabon':'GAB',
'Gambia, The':'GMB','Georgia':'GEO','Germany':'DEU','Ghana':'GHA','Gibraltar':'GIB',
'Greece':'GRC','Greenland':'GRL','Grenada':'GRD','Guam':'GUM','Guatemala':'GTM',
'Guernsey':'GGY','Guinea-Bissau':'GNB','Guinea':'GIN','Guyana':'GUY','French Guyana':'GUY','Haiti':'HTI',
'Honduras':'HND','Heard and McDonald Islands':'HMD','Hong Kong':'HKG','Hungary':'HUN','Iceland':'ISL',
'India':'IND','Indonesia':'IDN','Iran':'IRN','Iraq':'IRQ','Ireland':'IRL','Isle of Man':'IMN',
'Israel':'ISR','Italy':'ITA','Jamaica':'JAM','Japan':'JPN','Jersey':'JEY','Jordan':'JOR',
'Kazakhstan':'KAZ','Kenya':'KEN','Kiribati':'KIR','Korea, North':'KOR','Korea, South':'PRK',
'South Korea':'PRK','North Korea':'KOR','Kosovo':'KSV','Kuwait':'KWT','Kyrgyzstan':'KGZ',
'Laos':'LAO','Latvia':'LVA','Lebanon':'LBN','Lesotho':'LSO','Liberia':'LBR','Libya':'LBY','Liechtenstein':'LIE',
'Lithuania':'LTU','Luxembourg':'LUX','Macau':'MAC','Macedonia':'MKD','Madagascar':'MDG',
'Malawi':'MWI','Malaysia':'MYS','Maldives':'MDV','Mali':'MLI','Malta':'MLT','Marshall Islands':'MHL',
'Martinique (French)':'MTQ','Mauritania':'MRT','Mauritius':'MUS','Mexico':'MEX','Micronesia, Federated States of':'FSM',
'Moldova':'MDA','Moldavia':'MDA','Monaco':'MCO','Mongolia':'MNG','Montenegro':'MNE','Montserrat':'MSR',
'Morocco':'MAR','Mozambique':'MOZ','Myanmar':'MMR','Namibia':'NAM','Nepal':'NPL','Netherlands':'NLD',
'Netherlands Antilles':'ANT','New Caledonia':'NCL','New Caledonia (French)':'NCL','New Zealand':'NZL','Nicaragua':'NIC',
'Nigeria':'NGA','Niger':'NER','Niue':'NIU','Northern Mariana Islands':'MNP','Norway':'NOR','Oman':'OMN',
'Pakistan':'PAK','Palau':'PLW','Panama':'PAN','Papua New Guinea':'PNG','Paraguay':'PRY','Peru':'PER',
'Philippines':'PHL','Pitcairn Island':'PCN','Poland':'POL','Polynesia (French)':'PYF','Portugal':'PRT',
'Puerto Rico':'PRI','Qatar':'QAT','Reunion (French)':'REU','Romania':'ROU','Russia':'RUS','Russian Federation':'RUS',
'Rwanda':'RWA','Saint Kitts and Nevis':'KNA','Saint Lucia':'LCA','Saint Martin':'MAF','Saint Pierre and Miquelon':'SPM',
'Saint Vincent and the Grenadines':'VCT','Saint Vincent & Grenadines':'VCT','S. Georgia & S. Sandwich Isls.':'SGS','Samoa':'WSM',
'San Marino':'SMR','Saint Helena':'SHN','Sao Tome and Principe':'STP','Saudi Arabia':'SAU','Senegal':'SEN',
'Serbia':'SRB','Seychelles':'SYC','Sierra Leone':'SLE','Singapore':'SGP','Sint Maarten':'SXM',
'Slovakia':'SVK','Slovak Republic':'SVK','Slovenia':'SVN','Solomon Islands':'SLB','Somalia':'SOM','South Africa':'ZAF',
'South Sudan':'SSD','Spain':'ESP','Sri Lanka':'LKA','Sudan':'SDN','Suriname':'SUR',
'Swaziland':'SWZ','Sweden':'SWE','Switzerland':'CHE','Syria':'SYR','Taiwan':'TWN','Tajikistan':'TJK',
'Tadjikistan':'TJK','Tanzania':'TZA','Thailand':'THA','Timor-Leste':'TLS','Togo':'TGO','Tonga':'TON',
'Trinidad and Tobago':'TTO','Tunisia':'TUN','Turkey':'TUR','Turkmenistan':'TKM','Tuvalu':'TUV',
'Uganda':'UGA','Ukraine':'UKR','United Arab Emirates':'ARE','United Kingdom':'GBR','United States':'USA',
'U.S. Minor Outlying Islands':'UMI','Uruguay':'URY','Uzbekistan':'UZB','Vanuatu':'VUT',
'Vatican City State':'VAT','Venezuela':'VEN','Vietnam':'VNM','Virgin Islands':'VGB',
'Virgin Islands (USA)':'VIR','Virgin Islands (British)':'VGB','West Bank':'WBG','Yemen':'YEM',
'Zaire':'ZAR','Zambia':'ZMB','Zimbabwe':'ZWE'}