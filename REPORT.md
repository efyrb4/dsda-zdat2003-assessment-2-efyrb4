# Report

## Contents
- [California Housing Dataset](#california-housing-dataset)

- [Supervised ML](#supervised-ml)

  - [Random Forest Regression](#random-forest-regression)
  - [Random Forest Regression Experiments](#random-forest-regression-experiments)

    - [Random Forest Regression Experiment 1](#random-forest-regression-experiment-1)
      - [Random Forest Regression Experiment 1 Description](#random-forest-regression-experiment-1-description)
      - [Random Forest Regression Experiment 1 Results](#random-forest-regression-experiment-1-results)
      - [Random Forest Regression Experiment 1 Discussion](#random-forest-regression-experiment-1-discussion)

    - [Random Forest Regression Experiment 2](#random-forest-regression-experiment-2)
      - [Random Forest Regression Experiment 2 Description](#random-forest-regression-experiment-2-description)
      - [Random Forest Regression Experiment 2 Results](#random-forest-regression-experiment-2-results)
      - [Random Forest Regression Experiment 2 Discussion](#random-forest-regression-experiment-2-discussion)

    - [Random Forest Regression Experiment 3](#random-forest-regression-experiment-3)
      - [Random Forest Regression Experiment 3 Description](#random-forest-regression-experiment-3-description)
      - [Random Forest Regression Experiment 3 Results](#random-forest-regression-experiment-3-results)
      - [Random Forest Regression Experiment 3 Discussion](#random-forest-regression-experiment-3-discussion)

    - [Model comparison & evaluation](#model-comparison--evaluation)

- [Unsupervised ML](#unsupervised-ml)

  - [K-means Clustering](#k-means-clustering)

  - [K-means Experiments](#k-means-experiments)

      - [K-means Experiment 1](#k-means-experiment-1)
         - [K-means Experiment 1 Description](#k-means-experiment-1-description)
         - [K-means Experiment 1 Results](#k-means-experiment-1-results)
         - [K-means Experiment 1 Discussion](#k-means-experiment-1-discussion)

      - [K-means Experiment 2](#k-means-experiment-2)
         - [K-means Experiment 2 Description](#k-means-experiment-2-description)
         - [K-means Experiment 2 Results](#k-means-experiment-2-results)
         - [K-means Experiment 2 Discussion](#k-means-experiment-2-discussion)

## California Housing Dataset

The [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) consists of 8 features and 1 target variable (visible in the [ML Experiments Notebook](../src/ML_experiments.ipynb)) describing different geographical locations within California and their respective median house prices.

By applying the Interquartile Range (**IQR**) method (**Q1/Q3 +- 1.5 * IQR**), all features excluding the median house value and longitude/latitude were pre-processed. This resulted in 16,840 retained instances.

## Supervised ML

### Random Forest Regression

[Random Forest Regression (Overview)](https://builtin.com/data-science/random-forest-algorithm)

In this AI and ML project, the [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) was utilised to predict California housing prices. 

The decision to employ the Random Forest algorithm to predict California housing prices, was due to the model’s capability to handle complex and non-linear relationships between features and the target variable. Furthermore, a beneficial characteristic of the Random Forest regressor is its ability to determine feature importance, which provides valuable insights into the best predictors of house prices.

---

### Random Forest Regression Experiments

**Parameters used:**

- **`n_estimators`**: This parameter determines the number of decision trees to be used in the forest.


- **`max_depth`**: This parameter controls the maximum depth of each decision tree in the forest.


- **`max_features`**: This parameter specifies the maximum number of features the Random Forest is allowed to consider when looking for the best split at each node.

These parameters were selected to strike a balance between model performance, computational efficiency.

---
### Random Forest Regression Experiment 1

#### Random Forest Regression Experiment 1 Description

*Table 1: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(1)  | 50           | 3         | 2            |


The primary goal of the initial experiment is to establish a performance baseline for future experiments. By employing a reduced number of trees, limiting the tree depth, and selecting a small set of features, the model's performance can be assessed.

- **`n_estimators = 50`**: Using 50 estimators allows for a comparison between a small number of trees and the larger parameters, providing insights into predictive performance with fewer trees.

- **`max_depth = 3`**: Opting for a lower depth provides an understanding of the typical decision process within a tree in the forest.

- **`max_features = 2`**: Setting the maximum number of features to 2 for node splitting enables more efficient modelling compared to using all available features.

---

#### Random Forest Regression Experiment 1 Results

*Table 2: Performance Metrics*

| Model   | R^2 Score | RMSE     |
|---------|-----------|----------|
| RFR(1)  | 0.448     | 0.780|   |


<br><br>

![Actual vs Predicted](img/rfr/Model_1-(ActualvsPredicted).png)

*Figure 1: Actual vs Predicted House Prices*
<br><br>

![Residual plot](img\rfr\Model_1-(Residual-plot).png)

*Figure 2: Residual plot*
<br><br>

![Feature importance](img\rfr\Model_1-(Feature-Importance).png)

*Figure 3: Feature importance*
<br><br>

![Decision Tree example](img\rfr\Model_1-(Example-Tree).png)

*Figure 4: Decision Tree example*

---

#### Random Forest Regression Experiment 1 Discussion

- The model achieved an R-squared score of 0.448 and RMSE of 0.780 (**Table 1**), indicating the can explain 44.8% of the variance in house prices based on the features provided. The high RMSE and low R^2 suggests significant variation between actual and predicted values, signalling poor model performance.

- The actual vs predicted plot (**Figure 1**) illustrates inaccurate predictions and underestimation, particularly for expensive California houses.

- The residual plot (**Figure 2**) supports the underestimation of expensive houses, due to the majority of residuals clustered above the centre line. The non-random scatter suggests great inconsistency in predictions.

- The most significant feature (**Figure 3**) influencing housing prices is the median income.

- The example tree (**Figure 4**) displays the decision process for one tree in the forest. The root node reflects average occupancy diverging into sub nodes such as house attributes, location, and median income.

- To enhance the model and improve prediction accuracy, an increase to all parameters is appropriate to model more complex relationships.

---

### Random Forest Regression Experiment 2

#### Random Forest Regression Experiment 2 Description

*Table 3: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(3)  | 200          | 9         | 6            |

Based on the previous experiment, an increase in parameters seems suitable to achieve a better predictive performance. 

- **`n_estimators = 200`**: Due to bagging, I anticipate the predictive performance to increase as a greater ensemble of trees should lead to more accurate predictions.

- **`max_depth = 9`**: Increasing the tree depth within the forest will hopefully faciliate better modelling between the features and housing prices.

- **`max_features = 6`**: Enabling the model to have 6 features to split the nodes, will result in more complex rules to achieve a better decision process.

---
#### Random Forest Regression Experiment 2 Results

*Table 4: Performance Metrics*

| Model   | R^2 Score | RMSE     |
|---------|-----------|----------|
| RFR(3)  |  0.760    | 0.514    |


<br><br>

![Actual vs Predicted](img/rfr/Model_3-(ActualvsPredicted).png)

*Figure 5: Actual vs Predicted House Prices*
<br><br>

![Residual plot](img\rfr\Model_3-(Residual-plot).png)

*Figure 6: Residual plot*
<br><br>

![Feature importance](img\rfr\Model_3-(Feature-Importance).png)

*Figure 7: Feature importance*
<br><br>

---
#### Random Forest Regression Experiment 2 Discussion
- The model achieved an R-squared score of 0.760 and RMSE of 0.514 (**Table 6**), reflecting the importance of a large tree ensemble and greater tree complexity.

- **Figure 5** showcases greater alignment between actual and predicted values, attributed to data points closely clustered around the line.

- **Figure 6** demonstrates a tendency towards a random scatter of residuals around zero, indicating consistency in the model's predictions.

- Consistent with prior experiments, median income emerges as the most influential feature on house prices (**Figure 7**).

- Overall, there is notable improvement in model performance attributed to an increase in estimators in and tree complexity in the forest. 
---
### Random Forest Regression Experiment 3

#### Random Forest Regression Experiment 3 Description

*Table 5: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(4)  | 500          | 21        | 2            |

This final experiment seeks to determine the most optimal parameters ([Grid search](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)) for the Random Forest Regressor to achieve the best R^2, RMSE scores. 

- **`n_estimators = 500`**: 500 estimators in the forest yielded the best predictive capability among the tested parameters.

- **`max_depth = 21`**: The optimal tree depth indicates greater tree complexity enhances predictive capabilities.

- **`max_features = 2`**: The optimal number of features required to split a node was found to be two, suggesting that a combination of several features may not be the most effective in achieving accurate predictions.

---
#### Random Forest Regression Experiment 3 Results

*Table 6: Performance Metrics*

| Model   | R^2 Score | RMSE     |
|---------|-----------|----------|
| RFR(4)  |  0.804    | 0.465    |


<br><br>

![Actual vs Predicted](img/rfr/Model_4-(ActualvsPredicted).png)

*Figure 8: Actual vs Predicted House Prices*
<br><br>


![Residual plot](img\rfr\Model_4-(Residualplot).png)

*Figure 9: Residual plot*
<br><br>

![Feature importance](img\rfr\Model_4-(Feature-Importance).png)

*Figure 10: Feature importance*
<br><br>

---
#### Random Forest Regression Experiment 3 Discussion
The model achieved an R-squared score of 0.804 and RMSE of 0.465 (**Table 6**), suggesting an  increase in the number of estimators and tree complexity leads to an improved predictive capability.

- **Figure 8** reveals a close alignment between actual and predicted values, demonstrating strong predictive capability.

- **Figure 9** shows consistent predictions with a random scatter around zero, but extreme outliers affecting the overall model performance.

- Notably, the most important features are median income, location, and average occupancy, suggesting a deeper understanding of factors influencing housing prices (**Figure 10**).

- Overall, the experiments indicate that greater complexity and a larger ensemble of trees generally improves model performance, albeit with increased computational demands.
---

### Model comparison & evaluation

![Number_of_estimators](img\rfr\Model_performance-(Number_of_estimators).png)

*Figure 11: Model performance based on the number of estimators*
<br><br>


![Max_depth](img\rfr\Model_performance-(Max_tree_depth).png)

*Figure 12: Model performance based on the maximum tree depth*
<br><br>

![Model Comparison](img\rfr\Model_performance-Comparison.png)

*Figure 13: Model Comparison*
<br><br>

![Learning curve](img\rfr\Learning_curve.png)

*Figure 14: Learning curve*
<br><br>

- **Figures 11 and 12** illustrate the performance improvements with increasing parameters, supported by a model comparison in **Figure 13** showcasing changes across all models.

- **Figure 14** presents a learning curve, analysing the impact of training data on the validation set. Initially, the model shows signs of overfitting, but as training samples increase the validation performance improves.
---
## Unsupervised ML

### K-means Clustering
[K-means clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html) was employed to group median housing prices within California based on the underlying features such as location and house attributes. 

### K-means Experiments

**Parameters used:**

- **`n_clusters`**: This parameter defines the number of clusters for the clustering process.

---
### K-means Experiment 1

#### K-means Experiment 1 Description

![Optimal clusters](img\clustering\Optimal_clusters.png)
*Figure 15: Inertia and silhoutte plot to find optimal clusters*
<br><br>

By assessing **Figure 15** and evaluating the inertia and silhouette scores, the number of suitable clusters to represent housing prices for California's districts can be identified.

- **`n_clusters = 3`**: Based on **Figure 15**, 3 clusters strikes a balance between inertia and silhouette scores


#### K-means Experiment 1 Results

![Avg silhoutte plot](img\clustering\3_clusters_silhoutte.png)
*Figure 16: Silhoutte score per cluster*
<br><br>

![Geographical clusters](img\clustering\3_cluster_map.png)

*Figure 17: Geographical clusters based on Median house value*
<br><br>

![Actual clusters](img\clustering\Actual_Clusters.png)

*Figure 18: Actual geographical Median house value*
<br><br>


#### K-means Experiment 1 Discussion
- **Figure 16** displays the silhouette scores for each cluster, with larger coefficients reflecting better cohesion to other instances within the cluster. All clusters possess a greater than average silhouette coefficient, suggesting definitive clusters were created. Cluster 2 has the highest silhouette coefficient whilst clusters 1,0 possess wider characteristics suggesting more variance between instances in terms of housing prices.

- **Figure 17** depicts a geographical plot of the clusters across California. In comparison to **Figure 18**, the algorithm has identified that more expensive housing is concentrated along the California coast which is consistent with the actual districts. Cluster 1 covers the majority of California, highlighting the high variance among instances and the challenge of separating clusters 1 and 0.


#### K-means Experiment 2 Description
Based on the previous experiment, increasing the number of clusters appears appropriate due to the diificulty of classifying less expensive houses in California.

- **`n_clusters = 4`**: As 3 clusters provided close distinct groupings, a slight increment seems suitable to further separate the more varied clusters.

#### K-means Experiment 2 Results

![Avg silhoutte plot](img\clustering\4_cluster_silhoutte.png)
*Figure 19: Average silhoutte per cluster*
<br><br>

![Geographical clusters](img\clustering\4_cluster_map.png)

*Figure 20: Geographical clusters based on Median house value*
<br><br>

![Actual clusters](img\clustering\Actual_Clusters.png)

*Figure 21: Actual geographical Median house value*
<br><br>

#### K-means Experiment 2 Discussion

- The increase in clusters has led to a reduction in the variance within clusters with cluster 1 now being the most cohesive (**FIgure 19**). Although the clusters possess strong silhouette scores, there is still some variance between instances in cluster 0 which reflects the difficulty of clustering the lower prices houses in California.

- **Figure 20** shows similarities with the initial clusters in **Figure 21**. The model performs well in identifying the more expensive housing along the California coast and exhibits improvement in recognising lower housing prices, particularly along the southern coast. However, the model doesn't entirely capture the complexities of determining housing prices for the districts.

- Therefore, for further analysis, it might be beneficial to explore another model such as DBSCAN to cluster locations based on the median house value.






