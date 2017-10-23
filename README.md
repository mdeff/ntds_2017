# A Network Tour of Data Science, edition 2017

This repository contains the material for the labs associated with the EPFL
master course [EE-558 A Network Tour of Data Science][epfl] ([moodle]), taught
in autumn 2017. Compared to the [2016 edition], the course has been refocused
on graph and network sciences. The course material revolves around the
following topics:

1. [Network Science](https://en.wikipedia.org/wiki/Network_science),
1. [Spectral Graph Theory](https://en.wikipedia.org/wiki/Spectral_graph_theory),
1. [Graph Signal Processing](https://arxiv.org/abs/1211.0053),
1. [Machine Learning](https://en.wikipedia.org/wiki/Machine_learning).

[epfl]: http://edu.epfl.ch/coursebook/en/a-network-tour-of-data-science-EE-558
[moodle]: http://moodle.epfl.ch/course/view.php?id=15299
[2016 edition]: https://github.com/mdeff/ntds_2016

Below is the material you'll find in that repository:
1. [Practical informations][practical_info]
1. [Installation instructions](#installation)
1. [Introduction][d01]: conda & Anaconda, Python, Jupyter, git, scientific Python
1. Network properties: [twitter demo][d02], [numpy demo][d03], [assignment][a01], solution
1. Network models: [networkx demo][d04], assignment, solution
1. Spectral graph theory: demo, assignment, solution
1. Graph signal processing: demo, assignment, solution
1. Machine learning: demo, assignment, solution

As a Data Science course, the above activities are realized on real networks,
e.g. a social network from Twitter, that students have to collect and clean
themselves.

[practical_info]: https://github.com/mdeff/ntds_2017/raw/outputs/practical_info/ntds_labs.pdf

[d01]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/01_introduction.ipynb
[d02]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/02_data_acquisition_twitter.ipynb
[d03]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/03_numpy.ipynb
[d04]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/04_networkx.ipynb

[a01]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/assignments/01_network_properties.ipynb

## Installation

For these labs we'll need [git], [Python], and packages from the [Python
scientific stack][scipy]. If you don't know how to install those on your
platform, we recommend to install [Anaconda], a distribution of the [conda]
package and environment manager. Please follow the below instructions to
install it.

1. Open <https://www.anaconda.com/download> and download the Python 3.x
   installer for Windows, macOS, or Linux.
1. Install with default options.
1. Open the Anaconda Prompt (e.g. from the Windows Start menu).
1. Execute `conda config --add channels conda-forge` to add the [conda-forge]
   channel. It provides many additional packages to conda.
1. Install git with `conda install git`.
1. Download this repository by executing
   `git clone https://github.com/mdeff/ntds_2017`.
1. Open the Anaconda Navigator (e.g. from the Windows Start menu).
1. From there you can start e.g. Jupyter or Spyder.

[git]: https://git-scm.com
[python]: https://www.python.org
[scipy]: https://www.scipy.org
[anaconda]: https://anaconda.org
[conda]: https://conda.io
[conda-forge]: https://conda-forge.org
