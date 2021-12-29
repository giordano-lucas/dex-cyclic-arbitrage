<img width="1665" alt="Screenshot 2021-12-29 at 09 52 48" src="https://user-images.githubusercontent.com/43466781/147644243-f21b52b6-bb45-42fe-848f-7c7f1222646c.png">


DEX Cyclic Arbitrage Analysis

======

## Abstract

While the paper establishes the validity of the Tesco 1.0 dataset, we propose to use this dataset to study XXXXXXXX. To do so, we will study the clusters of XXXX computed out of their XXXX : areas with similar typical product consumption will be clustered together. And then, we will compare these clusters with XXXX. 


Such a clustering could help grocery shop companies to adapt their product according to the areas where they operate. Moreover, to provide a better understanding of the data we will create an interactive visualization. The latter will represent typical food consumption of London areas on a map of the city with different levels of aggregation, over different periods of time and with the possibility of choosing different attributes of the typical product. This visualisation will help assess the validity of our findings. 

Finally, we will spend some time analysing the actual output of the clustering in terms of the typical product of the clusters. Using the dataset of diabetes prevalence, we will try to quantify how much information is contained in the clusters and how usefull can it be.

E.g. study the impact of the cluster assignment as features of a linear regression compared to a base regressor.

## Research questions

1. 

Are similar areas (in terms of typical products) geographically close?
2. Can we naturally cluster areas geographically when it comes to food consumption in London?
3. How do those clusters differ when we let the aggregation level vary?
4. Is it possible to use techniques similar to those presented in the Tesco paper to validate the clustering results? Does the clustering contain valid and usable information?
5. In the context of an advertising campaign, which locations should be associated with which kinds of products?

## Proposed datasets

1. [Cyclic transaction dataset](https://disco.ethz.ch/misc/uniswap/cycles_in_Uniswap.json) -- This dataset contains all the necessary information to create the clusters described above. 
2. [Uniswap rates 1.0](https://drive.google.com/drive/folders/19mY0rxtHkAXRuO3O4l__S2Ru2YgcJVIA) --  This is the dataset used in the course to perform the first replication.


## Methods :

1. For the visualization task, we will use the library ```geopandas``` and ```bokeh```. In this visualization, we will include a button and checkboxes for the time (month) variable of the represented data,  the aggregation (ward,lsoa,msoa) level, and the feature of the typical product that will allow the user to visualize only a subset of the dataset. 
2. For the clustering part, we will cluster the areas according to the product consumption of their products using the two following methods:
  2.1. Apply the k means algorithm on the data (need also to find suitable k)
  2.2. Try to geographically validate the clustering using the following methods:
    * Use techniques seen in class to select the right value of ```k``` and  use dimensionality reduction to visualize the clusters in 2D
    * Find relevant geographical related metrics to evaluate the good-ness of fit. They will help us find formal evidence that there is (or not) a relation between geography and the clustering. For instance, we could use different distance metrics when the silhouette score is computed. We also plan to use a graph-based approach to compute such metrics (vertices are areas and edges link two areas if they share a physical border). 
3. Regarding the analysis of the clustering output, the following methods will be applied:
  3.1. For each cluster compute its typical product (average of the typical product of the areas contained in the cluster) and study the differences observed. We will then relate those differences in terms of the metabolic syndrome-related to diabetes prevalence (found in the Tesco paper).
  3.2. Quantify the predictive power of the clustering assignments to assess the information contained in the clustering. In order to do so, we will replicate the regression model (table 2. of the paper) on the number of diabetes prevalence in London and analyse the improvements (```R2``` for instance) when the clusters are added as dependent variable to the model compared to the base model of the paper. 
4. Conclude by proposing other possible practical usages of the clustering output.

## Timeline and contributions :

### Week 1 : Data Acquisition & Setup (1)

| Task                                                                    | Team member(s)                  | work hours  |
| :-----------------------------------------------------------------------|:--------------------------------| -----------:|
| Literature research                                                     | Lucas & Augustin                | 3h          |
| API choice and query design                                             | Lucas & Augustin                | 4h          |
| EPFL Cluster & environment setup                                        | Lucas                           | 2h          |
| Data fetching script test                                               | Augustin                        | 3h          |
| Data fetching validation                                                | Augustin                        | 1h          |
| Data fetching improvements                                              | Augustin                        | 2h          |


### Week 2 : Data preprocessing

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Data cleaning                            | Augustin                        | 4h          |
| Data exploration                         | Lucas                           | 1h          |
| Raw data => embedding format             | Lucas                           | 3h          |


### Week 3 : Embedding & Clustering 

| Task                                     | Team member(s)                  | work hours  |
| :----------------------------------------|:--------------------------------| -----------:|
| Autencoder keras basic code              | Lucas                           | 2h          |
| Comparision with PCA and debugging       | Lucas                           | 1h          |
| K-means                                  | Augustin                        | 1h          |

### Week 4 : Clustering analysis, Profitablity prediction & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Clustering analysis                     | Lucas                           | 4h          |
| Profitablity prediction setup           | Augustin                        | 2h          |
| Github pages setup                      | Lucas                           | 2h          |
| Data story (1)                          | Lucas                           | 3h          |

### Week 5 : Hyperparameter opmisation and improvements & report writing

| Task                                    | Team member(s)                  | work hours  |
| :---------------------------------------|:--------------------------------| -----------:|
| Hyperparameter opmisation               | Lucas & Augustin                | 2h          |
| Notebook comments and markdown          | Lucas & Augustin                | 3h          |
| Data story (2)                          | Lucas & Augustin                | 6h          |

### Total contribution:

| Team member                     | work hours   |
|:--------------------------------| ------------:|
| Lucas Giordano                  | 38h          |
| Lucas Gruaz                     | 24h          |
| Augustin Kapps                  | 35h          |


## Notes to the reader
### Organisation of the repository

In order to be able to run our notebook, you should have a folder structure similar to:

    .
    ├── data                                      # Data folder
    │ ├── diabetes_estimates_osward_2016.csv      # [Diabetes estimates Osward](https://drive.google.com/drive/folders/19mY0rxtHkAXRuO3O4l__S2Ru2YgcJVIA)
    │ ├── all                                     # folder containing [Tesco grocery 1.0](https://figshare.com/articles/Area-level_grocery_purchases/7796666)
    │ │  ├── Apr_borough_grocery.csv              # example of file
    │ │  ├── ...
    │ ├── statistical-gis-boundaries-london       # folder containing the unzipped [Statistical GIS Boundary Files for London](https://data.london.gov.uk/dataset/statistical-gis-boundary-files-london) 
    │ │  ├── ESRI                                 # contains the data to be loaded by geopandas
    │ │  │  ├── London_Borough_Excluding_MHW.dbf  # example of file
    │ │  │  ├── ...
    │ │  ├── ...
    ├── images                              # Contains the ouput images and html used for the data story
    ├── extension.ipynb                     # Deliverable notebook for our extension
    ├── vizu.ipynb                          # Notebook containing only the vizualisations (if the reader only was to see the interactive viz)
    ├── Data Extraction.ipynb               # Notebook that generates the subset of tesco used in this analysis
    └── README.md               
    
Regarding the data folder a zip file can be downloaded [here](https://drive.google.com/drive/folders/1DH7EXo6Pbm2guJkWW75-wbYPa_5KTGQd?usp=sharing). It only remains to place it under the root directory of the repository and unzip it to be able to run the following notebooks. 

> ```extension.ipynb``` is therefore the "single" notebook that we are supposed to deliver for the P4 submission.

Note: ```vizu.ipynb``` is only there for the people who read our data story but still want to play with the interactive vizualisation without having to go through the entire deliverable notebook. Furthermore, in the previously mentioned drive folder, the zip should contain a file named ```tesco.csv```. The latter contains all the revelant information from the original ```Tesco``` dataset for this analysis. It is nicely indexable and makes easier for us to query information. Alternatively, you could simply run the notebook ```Data Extraction.ipynb``` to generate it.

### Dependencies requirement

In the repository, we provide a `requirement.txt` file from which you can create a virutatal python environment.
