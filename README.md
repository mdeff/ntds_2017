# A Network Tour of Data Science, edition 2017

[![Binder](https://mybinder.org/badge.svg)][binder_lab]
&nbsp; (Jupyter [lab][binder_lab] or [notebook][binder_notebook])

[binder_lab]: https://mybinder.org/v2/gh/mdeff/ntds_2017/outputs?urlpath=lab
[binder_notebook]: https://mybinder.org/v2/gh/mdeff/ntds_2017/outputs?urlpath=tree

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
1. Network properties: [twitter demo][d02], [numpy demo][d03], [assignment][a01],
   [solution][a01s1], [student solution][a01s2], [feedback][a01fb]
1. Network models: [networkx demo][d04], [matplotlib demo][d05], [assignment][a02], solution
1. Spectral graph theory: [web API and pandas demo][d06], [assignment][a03], solution
1. [Data exploration and visualization demo][d07]
1. Graph signal processing: [demo][d08], [assignment][a04], solution

As a Data Science course, the above activities are realized on real networks,
e.g. a social network from Twitter, that students have to collect and clean
themselves.

[practical_info]: https://github.com/mdeff/ntds_2017/raw/outputs/practical_info/ntds_labs.pdf
[projects]: https://github.com/mdeff/ntds_2017/raw/outputs/projects/ntds_projects.pdf

[d01]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/01_introduction.ipynb
[d02]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/02_data_acquisition_twitter.ipynb
[d03]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/03_numpy.ipynb
[d04]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/04_networkx.ipynb
[d05]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/05_matplotlib.ipynb
[d06]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/06_webapi_pandas.ipynb
[d07]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/07_data_exploration_and_visualisation.ipynb
[d08]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/demos/08_pygsp.ipynb

[a01]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/assignments/01_network_properties.ipynb
[a01s1]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/assignments/01_solution_ersi.ipynb
[a01s2]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/assignments/01_solution_florian.ipynb
[a01fb]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/assignments/01_feedback.ipynb
[a02]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/assignments/02_network_models.ipynb
[a03]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/assignments/03_spectral_graph_theory.ipynb
[a04]: https://nbviewer.jupyter.org/github/mdeff/ntds_2017/blob/outputs/assignments/04_graph_signal_processing.ipynb

## Projects

Part of the course is evaluated by a project (see the [description][projects]),
proposed and carried out by groups of three to four students. Below is their
work.

## Usage

Click the [binder badge][binder_lab] to play with the notebooks from your
browser without installing anything.

For a local installation, you will need [git], [Python], and packages from the
[Python scientific stack][scipy]. If you don't know how to install those on
your platform, we recommend to install [Miniconda], a distribution of the
[conda] package and environment manager. Please follow the below instructions
to install it and create an environment for the course.

1. Download the Python 3.x installer for Windows, macOS, or Linux from
   <https://conda.io/miniconda.html> and install with default settings. Skip
   this step if you have conda already installed (from [Miniconda] or
   [Anaconda]). Linux users may prefer to use their package manager.
   * Windows: Double-click on the `.exe` file.
   * macOS: Run `bash Miniconda3-latest-MacOSX-x86_64.sh` in your terminal.
   * Linux: Run `bash Miniconda3-latest-Linux-x86_64.sh` in your terminal.
1. Open a terminal. Windows: open the Anaconda Prompt from the Start menu.
1. Install git with `conda install git`.
1. Download this repository by running
   `git clone https://github.com/mdeff/ntds_2017`.
1. Create an environment with the packages required for the course with
   `conda env create -f ntds_2017/environment.yml`.

Every time you want to work, do the following:

1. Open a terminal. Windows: open the Anaconda Prompt from the Start menu.
1. Activate the environment with `conda activate ntds_2017`
   (or `activate ntds_2017`, or `source activate ntds_2017`).
1. Start Jupyter with `jupyter notebook` or `jupyter lab`. The command should
   open a new tab in your web browser.
1. Edit and run the notebooks from your browser.

[git]: https://git-scm.com
[python]: https://www.python.org
[scipy]: https://www.scipy.org
[anaconda]: https://anaconda.org
[miniconda]: https://conda.io/miniconda.html
[conda]: https://conda.io
[conda-forge]: https://conda-forge.org
