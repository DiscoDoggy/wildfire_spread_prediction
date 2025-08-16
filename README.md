# Next Day Wildfire Spread
## Key Question:
How well can Machine Learning methods predict how a wildfire may spread?
## Dataset:
This work follows Fantine Huot et al’s “Next Day Wildfire Spread” dataset and was first published in IEEE Transactions on Geoscience and Remote Sensing in a paper titled “Next Day Wildfire Spread: A Machine Learning Dataset to Predict Wildfire Spreading From Remote-Sensing Data”.
In this paper Huot develops a remote sensing dataset and shows that machine learning models can indeed predict next day wildfire spread to some degree. The dataset can be found on Kaggle as well. 

The data is stored as .tfrecords. There are 19 tfrecords: 15 training, 2 eval, and 2 test. Each tfrecord can contain up to 1000 samples. In total, there are about 18,545 fire events but to save time and memory only about 10000 are used during training. 

Each data sample is a set of remote sensing features, a current day wildfire “map”, and the ground truth next day wildfire “map”. Each data sample represents these features over a 64km by 64km area. However, the data is not explicitly stored like a map or 2d image.  Tfrecord samples are readable as key value pairs where for a specific sample printing the feature yields 4096 values (each square kilometer inside a 64km x 64km area). 

An example of what one feature for one sample is serialized as is shown below:

<img width="124" height="589" alt="image" src="https://github.com/user-attachments/assets/5dbdc1ff-1dba-4546-936d-b9beca71bfcf" />

Variable Topic
Description
Elevation
Heatmap detailing elevation over a region in meters
Wind direction
Heatmap detailing wind direction in degrees over a region
Wind speed
Heatmap detailing speed of wind in meters per second over a region
Min temp.
Heatmap detailing the minimum temperature of a series of sampled temperatures over the course of a few hours over a certain region in Kelvin
Max Temp
Heatmap detailing the maximum temperature of a series of sampled temperatures over the course of a few hours over a certain region in Kelvin
Humidity
Heatmap detailing the recorded humidity over a region in kg/kg which is ratio of mass of water vapor to the mass of dry air
Precipitation
Heatmap detailing the precipitation over a region in millimeters. Recorded precipitation is defined as the daily total.
Drought Index
Heatmap detailing the drought index over an area. Measured from 2.0+ being very wet to -2 or less being extremely dry
Vegetation
Heatmap of vegetation over an area measured through Normalized Difference Vegetation Index (NDVI). Describes greenness and density of vegetation
Population Density
Heatmap of population over a region measured as people per square kilometer. 84% of fires caused by humans. This feature is meant to capture that likelihood.
Energy Release Component
Output of a national fire danger rating system describing the relationship between fuel moisture and potential fire intensity
Previous Fire mask
Fire mask over a region at time step t which is the current day
Fire mask
Fire mask over a region at time step t + 1 this is the target variable

