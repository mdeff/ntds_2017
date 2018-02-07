# Titanic

Building on the top of recent advances in the field of signal processing on graphs (Schuman et al., 2013) and deep learning on irregular domains (Bronstein et al., 2017), we investigate the performance of standard machine learning methods and the relevance of graph based convolutional neural networks to perform binary classification in this specific case (layered data). The new method provide a convenient way of getting rotational invariance over the data (Defferrard et al., 2017) and set up a flexible framework for structured pooling.

See the [project report](./project.ipynb) (notebook).

## Getting started

```shell
7z e data.7z 
pip3 install -r requirements.txt
jupyter notebook
# or
jupyter lab
```

## References

- TORRES, Ramon, SNOEIJ, Paul, GEUDTNER, Dirk, et al. GMES Sentinel-1 mission. Remote Sensing of Environment, 2012, vol. 120, p. 9-24.
- SHUMAN, David I., NARANG, Sunil K., FROSSARD, Pascal, et al. The emerging field of signal processing on graphs: Extending high-dimensional data analysis to networks and other irregular domains. IEEE Signal Processing Magazine, 2013, vol. 30, no 3, p. 83-98.
- BRONSTEIN, Michael M., BRUNA, Joan, LECUN, Yann, et al. Geometric deep learning: going beyond euclidean data. IEEE Signal Processing Magazine, 2017, vol. 34, no 4, p. 18-42.
- DEFFERRARD, Michaël, BRESSON, Xavier, et VANDERGHEYNST, Pierre. Convolutional neural networks on graphs with fast localized spectral filtering. In : Advances in Neural Information Processing Systems. 2016. p. 3844-3852.
- NGUYEN Ha Q., DO Minh N, et al. Downsampling of Signal on Graphs Via Maximum Spanning Trees. IEEE Transactions on Signal Processing, 2015, vol. 63, no 1.
- DORFLER Florain, BULLO Francesco. Kron reduction of graphs with applications to electrical networks. 2011.

## License

The project is licensed under MIT. Datasets follow their respective licensing schemes and are not assimilated to the processing. Some functions are courtesy of [Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering](https://github.com/mdeff/cnn_graph) (MIT) or adapted from [PyGSP: Graph Signal Processing in Python](https://github.com/epfl-lts2/pygsp/) (BSD).
