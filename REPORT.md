# Report

## Contents

- [Supervised ML](#supervised-ml)
  - [Random Forest Regressor](#random-forest-regressor)
  - [Supervised ML Dataset](#supervised-ml-dataset)
  - [Supervised ML Experiments](#supervised-ml-experiments)
    - [Supervised ML Experiment 1](#supervised-ml-experiment-1)
    - [Supervised ML Experiment 1 Description](#supervised-ml-experiment-1-description)
    - [Supervised ML Experiment 1 Results](#supervised-ml-experiment-1-results)
    - [Supervised ML Experiment 1 Discussion](#supervised-ml-experiment-1-discussion)
- [Unsupervised ML](#unsupervised-ml)


## Supervised ML

### Random Forest Regressor

[Random Forest Regressor (Sklearn)](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

[Random Forest Regression (Overview)](https://builtin.com/data-science/random-forest-algorithm)

In this AI and ML project, the **Random Forest Regressor** was utilised to predict California housing prices. The Random Forest algorithm is an ensemble method that combines numerous decision trees and aggregates all predictions from the trees to achieve more accurate results (**bagging**). Through combining multiple trees and **bootstrapping**, the model develops an ability to generalise well to new samples whilst mitigating the risk of overfitting to the training data. 

The decision to employ the Random Forest algorithm to predict California housing prices, was due to the model’s capability to handle complex and non-linear relationships between features and the target variable. Features such as location, population, and house age may not exhibit linear relationships with housing prices, making models such as linear regression unsuitable for accurate predictions.

Furthermore, the ensemble of trees enables the handling of extreme outliers or anomalies within the housing market as the model typically can adjust well to unseen samples. Finally, a beneficial characteristic of Random Forest regression is feature importance, which provides valuable insights into the best predictors of house prices.



### Supervised ML Dataset

A very brief description of the dataset used for this technique. The datasets should not require any signifiant pre-processing to use with the model, so you can link to the documentation for the dataset here and provide only changes you have made or your usage.

[California Housing dataset (Sklearn)](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

### Summary of California Housing Dataset:

The California Housing dataset contains features describing various geographical locations in California and targets the median house value for districts. Each data instance represents a district, and the dataset includes the following features:



### Summary:
The California Housing dataset offers opportunities for both supervised and unsupervised machine learning techniques. Supervised learning can be applied to predict house prices based on demographic and geographical features, while unsupervised learning can provide insights into spatial patterns, regional disparities, and anomalies within California districts. These techniques can inform urban planning, real estate development, and policy-making decisions by understanding housing market dynamics and socio-economic patterns across different regions of California.

### Supervised ML Experiments

**Repeat the following sections for each experiment you run.**

#### Supervised ML Experiment 1

##### Supervised ML Experiment 1 Description

A description of how you experimented with the model (e.g. parameter tuning) and the goal of the experiment. You can refer to other files in your project, but you should include here things like a table containing the configuration you employed, what you changed, and why you changed it. What exactly you include here will depend on the technique you've selected and the experiments you choose to run.

#### Supervised ML Experiment 1 Results

The results of the experiment. This should include any graphs or tables that you have generated as well as any relevant metrics or statistics that you have calculated. You can include here qualitative observations, comparisons with a baseline/other parameters (if relevant), etc. Consider your presentation of results carefully to show concise but clear information without just bombarding the reader with plots and text!

#### Supervised ML Experiment 1 Discussion

Here, you should include an analysis and evaluation of the results of your experiments. This can include broader considerations than the results, such as: a discussion on the significance and implications of the results with respect to the data and, critically, the model itself; an assessment of the model's performance linked to the experiments; and any observed trends or anomalies; reflection on the experiment's limitations and potential biases; any limitations of the experiment and any potential improvements that could be made; etc.

## Unsupervised ML

For the unsupervised ML part of the report, you can follow the same structure as for the [supervised ML part](#supervised-ml) above.

### Application of Unsupervised Machine Learning:

1. **Clustering Analysis:**
   - Apply clustering algorithms such as K-means or DBSCAN to group similar geographical regions together based on demographic features.
   - Identify clusters of regions with similar housing characteristics, which can provide insights into spatial patterns and regional disparities.
   - Visualization techniques such as heatmaps or choropleth maps can be used to visualize cluster densities and spatial patterns of housing characteristics.

2. **Dimensionality Reduction:**
   - Use techniques like Principal Component Analysis (PCA) to reduce the dimensionality of the dataset and visualize high-dimensional data in lower dimensions.
   - Explore underlying patterns or relationships between features and identify the most informative features that distinguish between districts.

3. **Anomaly Detection:**
   - Identify outliers or unusual patterns in the data, such as districts with significantly different housing characteristics compared to neighboring regions.
   - Anomaly detection techniques can help identify regions with unexpected housing trends or anomalies in the data, which may require further investigation.
