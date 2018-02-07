# GraphLang

Reading is too mainstream. What if you could get the important ideas from a text without even reading it ? What about comparing several documents based on their textual content ? Or maybe you just want to visualize the concepts present in a book and their interaction ? GraphLang is the tool you need to boost your texts using graphs.

## How ?

This project is all about analysing textual resources using graphs and extracting useful insights. The main idea is to represent a document using the cooccurrences of its words, turning that into a graph and leverage the power of graph analysis tools in order to better understand the document. 

At first the graph could be built by only considering words directly adjacent to each other and representing this proximity with a link in the graph, where the nodes would be the words themselves. The recipe could then be complexified by considering also words at distance N from each other (N would have to be defined) and defining edge weights as a function of N. Punctuation could also be taken into account and would influence the weight of edges (two words, one at the end of a sentence and the other at the beginning of the next one shouldn’t (maybe) have a strong edge between them). This graph could be extended to take into account multiple documents at once using signals on the edges.

On the other hand, we could consider taking a set of documents and create for each document a set of features that characterizes the particularity of each document. Inspired by the homework 03, we could build a graph using those features and then, using spectral decomposition, we could represent each document in a reducted space.

[Grégoire CLEMENT](https://github.com/gregunz), [Maxime DELISLE](https://github.com/maxime-delisle), [Charles GALLAY](https://github.com/cgallay) and [Ali HOSSEINY](https://github.com/ali-h)

## Dependencies
In order to execute our notebook, you'll need the following libraries:
- numpy
- pandas
- networkx
- seaborn
- scikit-learn (sklearn)
- matplotlib
- tqdm
- plotly
- python-louvain (collections)
