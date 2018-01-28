# NTDS_Project

The data is on dropbox (NTDS_data folder accessible to  all contributors) : https://www.dropbox.com/home/NTDS_data

The report is contained in the notebook "A Network Tour Of Data Science project - Course Suggester- Report".

The project was conducted in multiple stages and so in different notebooks :

- Data Acquisition and cleaning : "Data Cleaning" notebook
- Data Exploration :
  - Creating the graphs : "Graph Creation", "Topics Graph", "Graph requirements", graph_module.py
  - Analysis : "Graph Analysis clean" notebook
- Data Exploitation : "Spectral clustering", "Weighting metrics and graph diffusion" notebooks

All what we did in the different notebooks and their respective results are covered in the report notebook. Thus, we chose to not have a main notebook but rather to have different notebooks corresponding to the different parts of the report. Therefore, the report notebook is sufficient to know what we did and which results we got, but you can go through the other notebooks which are commented and detailed if you want to know how we coded the different parts. In order to know which notebook corresponds to which part, we referenced them in the report.
¨
Executing the notebooks provided any modifications should be done in the order presented earlier in order to have the most up to date data.

In order to execute the notebooks, please respect the following architecture (similar to the one provided and the one on github https://github.com/LailaHms/NTDS_Project):

|Folder "data"		|data from dropbox
|Folder of the notebooks|folder "Graphs"|
			|Notebook1
			|Notebook2
			|...
