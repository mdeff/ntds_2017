# Community detection on the Wikipedia Hyperlink Graph
## NTDS project

Team members: 
* Armand Boschin
* Bojana Ranković
* Quentin Rebjock

### Important note:
Please take the time to download the pickle file `shortest_paths.pkl` from this [link](https://drive.google.com/file/d/17bXr-OKY8xrUhCDfwR0WwP9uTOXeKG9d/view?usp=sharing). It should be put in the directory `data/`.

In order to read the notebook properly (especially the graph visualization in the end), you might want to used this [link](https://nbviewer.jupyter.org/github/armand33/wikipedia_graph/blob/master/ntds_project.ipynb?flush_cache=true) to NBViewer.

### Content of the folder:
* ntds_project.ipynd : this is the main notebook containing all the data pipeline and the exploitation of the data
* utils.py : this file contains useful homemade Python functions that were used during the scraping and the exploitation of the
data.
* data : folder containing some pickle files
    * network.pkl : pickle file of the scraped network dictionary (see the notebook for details)
    * shortest_paths.pkl : pickle file of the shortest paths dictionary (see the notebook for details)
    * cross_val.pkl : pickle file on the cross-validation of the k parameter (see the notebook for more details)
    * viz.gephi : gephi file for the visualization of the communities
* images : screenshots of some visualization of the communities done using Gephi.
### Requirements:
This project was developed using Python 3.

The required libraries are: 
* Numpy
* NetworkX
* Scikit Learn
* community
* python-louvain
* seaborn
* tqdm
* plotly
 

### Project details
[Link](https://docs.google.com/document/d/1XEc3ogZWYKrAFKGfEoxY8xdoTWGB_P_4vfVQA0mi6tk/edit?usp=sharing) to the original 
project proposal.

Here is a reminder of the project proposal adapted to the reality of the project:

***Graph:*** Wikipedia hyperlink network

***Problem:*** 
Does the structure of the graph bears info on the content of the nodes ? We would like to find out if it is possible to detect communities of pages just by looking at the hyperlink connections and match these communities with real-world data such as categories of the pages. Is spectral clustering a viable possibility compared to proven method of community detection.

***Steps of the project:***
* Scraping the Wikipedia hyperlink network. Start from one node and get the pages as far as 2 or 3 hops depending on the number of nodes we get.
* Model the network by a random graph/scale-free network/something else in order to try to retrieve some of its characteristics.
* Apply Louvain algorithm for community detection to get a baseline to compare spectral clustering to.
* Try to apply spectral clustering in order to detect communities of pages.
* Visualize the clusters to match them with real-world categories.
