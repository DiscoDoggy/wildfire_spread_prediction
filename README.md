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

We provide a list of all features and their corresponding meaning below:

| Variable Topic | Description |
| :--- | :--- |
| Elevation | Heatmap detailing elevation over a region in meters |
| Wind direction | Heatmap detailing wind direction in degrees over a region |
| Wind speed | Heatmap detailing speed of wind in meters per second over a region |
| Min temp. | Heatmap detailing the minimum temperature of a series of sampled temperatures over the course of a few hours over a certain region in Kelvin |
| Max Temp | Heatmap detailing the maximum temperature of a series of sampled temperatures over the course of a few hours over a certain region in Kelvin |
| Humidity | Heatmap detailing the recorded humidity over a region in kg/kg which is ratio of mass of water vapor to the mass of dry air |
| Precipitation | Heatmap detailing the precipitation over a region in millimeters. Recorded precipitation is defined as the daily total. |
| Drought Index | Heatmap detailing the drought index over an area. Measured from 2.0+ being very wet to -2 or less being extremely dry |
| Vegetation | Heatmap of vegetation over an area measured through Normalized Difference Vegetation Index (NDVI). Describes greenness and density of vegetation |
| Population Density | Heatmap of population over a region measured as people per square kilometer. 84% of fires caused by humans. This feature is meant to capture that likelihood. |
| Energy Release Component | Output of a national fire danger rating system describing the relationship between fuel moisture and potential fire intensity |
| Previous Fire mask | Fire mask over a region at time step t which is the current day |
| Fire mask | Fire mask over a region at time step t + 1 this is the target variable |

## Visualizing the Data:
Now that we know how the data is serialized, we can utilize Python and Python's matplotlib to assemble 64x64 heatmaps of each feature. 64x64 where a single cell represents a 1km x 1km area.

<img width="1390" height="818" alt="image" src="https://github.com/user-attachments/assets/36352534-256e-4395-b5e6-066e901990a4" /> 

## Preprocessing the Data
We plan to take two approaches for wildfire spread prediction: traditional ML methods and Deep Learning. Each calls for different preprocessing methodologies. 

### Deep Learning Preprocessing
For deep learning, we aim to use convolutional neural networks so we can frame the problem as an image segmentation problem given a set of channels. Here each channel represents a 64x64 km feature map. To preprocess, we convert each feature for a particular sample
into a 64 x 64 "image" in which we stack on top of eachother to get a 64 x 64 x 12 training sample.

### Machine Learning Preprocessing
For machine learning, we cannot use the same approach as we did in deep learning. If we did, we would have a 49,152 feature sample which is far too many samples. Instead, we take s 3x3x12 windows and predict the center pixel as fire or no fire. The idea is that we can then string multiple of these predictions together to create a firemap. We cut 49,152 features down to 108 features (3x3x12).

## Machine Learning Results
### Logistic Regression

Logistic regression unsurprisingly underfits the data significantly. We opt not to use any regularization due to this fact. Logistic regression likely is not complex enough of a model to learn the underlying structures behind wildfire spread.
We analyze probability maps and observe that the model is far too overconfident about the presence of wildfire spread exemplified by entire fire maps being colored yellow.

<img width="1159" height="743" alt="image" src="https://github.com/user-attachments/assets/401f5e2a-53b1-44f0-be5a-ba1a512b68c0" />

### Support Vector Machines (SVMs)
We achieve marginally better results visually compared to logistic regression. Looking at fire map 1 and fire map 4, we see at least similar volumes of fire pixels predicted instead of the overwhelming mostly fire fire maps we see in logistic regression. Fire map 1 for SVMs understands the general location of some of the fires. Fire map 2,3, and 5 still largely over predict the presence of wildfires. 

<img width="1328" height="667" alt="image" src="https://github.com/user-attachments/assets/202840cb-57b7-4f0a-84ee-0c74d5154a66" />

### Random Forest
Random forest performs incredibly well and is also confident about its decisions. Random forest closely matches the number of fire pixel clusters along with the general location of where the clusters are. Additionally, compared to logistic regression, random forest has high confidence when it predicts a fire pixel and when there is no fire pixel there is very low confidence. The model still struggles when larger clusters are closer together, seeming to combine the clusters rather than separate. This could be due to the fact that the model only has 3x3x12 windows and each “pixel’s” prediction is independent of the previous. 

<img width="1204" height="771" alt="image" src="https://github.com/user-attachments/assets/7f6a63e3-9390-45ac-997e-04bde6061e52" />

### Gradient Boosted Trees
The gradient boosted trees perform similarly to random forest. They both get the general number of clusters and general location close to correct. The results for the gradient boosted trees differ from the random forest in the fact that the gradient boosted trees tend to have a noisier fire map. There are a few extra specs of fire pixels scattered across the entire map even when there are clusters in the correct location and size. The extra fire specs are interesting because the probability map shows that the model is not as confident about those specs as they are about the correctly classified clusters. Gradient boosted trees also tend to predict geometrically rounded shapes such as rectangles which is interesting.

<img width="1180" height="748" alt="image" src="https://github.com/user-attachments/assets/f2221031-e8f7-4e94-a455-91afdbbf282e" />

### Deep Learning - Convolutional Neural Network
Due to computational constraints, we implement a deep convolutional neural network and we also do not use very much of the training data. The convolutional neural network used is not a ResNet possibly explaining why the model shows little improvements between 100 epochs and 200 epochs. The combination of a deepish model with not using a lot of the training data could explain poor performance. Huot with a Resnet gets better results.

<img width="1314" height="567" alt="image" src="https://github.com/user-attachments/assets/819b6944-d28e-46c6-876b-d81de25154c3" />

<img width="1186" height="524" alt="image" src="https://github.com/user-attachments/assets/1649bbc8-71e3-4e28-b78e-1b9cd8583509" />


