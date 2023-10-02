# Apartment-Buddy
# Exploratory-Analysis-Of-Geolocational-Data
This project caters to a wide audience, including students and working professionals. It helps individuals find accommodation near their preferred food outlets, benefiting both residents and food providers. Restaurant managers can use this data to identify optimal locations based on customer demographics.

# The project consists of the following stages:
Data Acquisition: Gather datasets from appropriate sources.
Data Preprocessing: Prepare and clean the datasets for analysis using Pandas.
Data Visualization: Visualize the data through boxplots using Matplotlib, Seaborn, or Pandas.
Geospatial Data Retrieval: Fetch geolocational data from the REST APIs on the HERE API Website or others.
Clustering Analysis: Apply K-Means Clustering for location clustering using ScikitLearn.
Spatial Presentation: Displayed the results on a map using Folium.

![Flow Image](https://drive.google.com/file/d/1BZR8nC5ERaGFO6FUamLj2NTzRR6lPZx9/view?usp=drive_link)

# Result after implementation
![Output Image](https://drive.google.com/file/d/1yBKs2jF0ks538apepOT1gCiL1e958gxq/view?usp=drive_link)

# Conclusion

1. Green: Apartments in the cluster with index 0.
2. Orange: Apartments in the cluster with index 1.
3. Red: Apartments in the cluster with index 2.

These colors are used to visually differentiate between the clusters of apartments on the map. Each cluster represents a group of apartments that are similar in terms of their amenities and other factors, as determined by the K-Means clustering algorithm. The specific meaning of each cluster (e.g., proximity to amenities, budget, etc.) would depend on the characteristics of the data and the results of the clustering analysis.
