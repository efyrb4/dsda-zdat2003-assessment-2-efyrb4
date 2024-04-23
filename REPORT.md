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

    - [Random Forest Regression Experiment 4](#random-forest-regression-experiment-4)
      - [Random Forest Regression Experiment 4 Description](#random-forest-regression-experiment-4-description)
      - [Random Forest Regression Experiment 4 Results](#random-forest-regression-experiment-4-results)
      - [Random Forest Regression Experiment 4 Discussion](#random-forest-regression-experiment-4-discussion)

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

Before performing Random Forest Regression, I pre-processed the dataset to remove outliers within the features. By applying the Interquartile Range (**IQR**) method (**Q1/Q3 +- 1.5 * IQR**), all features excluding the median house value and longitude/latitude were pre-processed. This resulted in 16,840 retained instances.

## Supervised ML

### Random Forest Regression

[Random Forest Regression (Overview)](https://builtin.com/data-science/random-forest-algorithm)

In this AI and ML project, the [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) was utilised to predict California housing prices. The Random Forest algorithm is an ensemble method that combines numerous decision trees and aggregates all predictions from the trees to achieve more accurate results (**bagging**). Through combining multiple trees and **bootstrapping**, the model develops an ability to generalise well to new samples whilst mitigating the risk of overfitting to the training data. 

The decision to employ the Random Forest algorithm to predict California housing prices, was due to the model’s capability to handle complex and non-linear relationships between features and the target variable. Features such as location, population, and house age may not exhibit linear relationships with housing prices, making models such as linear regression unsuitable for accurate predictions. Furthermore, a beneficial characteristic of the Random Forest regressor is its ability to determine feature importance, which provides valuable insights into the best predictors of house prices.

---

### Random Forest Regression Experiments

**Parameters used:**

- **`n_estimators`**: This parameter determines the number of decision trees to be used in the forest.


- **`max_depth`**: This parameter controls the maximum depth of each decision tree in the Random Forest.


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

- **`n_estimators = 50`**: Using 50 estimators allows for a comparison between a small number of trees and the default parameters, providing insights into predictive performance with fewer trees.

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

- The residual plot supports the underestimation of expensive houses, due to the majority residuals clustered above the centre line. The non-random scatter suggests great inconsistency in predictions, reflecting poor model performance.

- The most significant feature (**Figure 3**) influencing housing prices is the median income.

- The example tree (**Figure 4**) displays the decision process for one tree in the forest. The root node reflects average occupancy diverging into sub nodes such as house attributes, location, and median income.

- To enhance the model and improve prediction accuracy, an increase to all parameters is appropiate to model more complex relationships.

---

### Random Forest Regression Experiment 2

#### Random Forest Regression Experiment 2 Description

*Table 3: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(2)  | 100           | 6        | 4           |

The objective of this experiment is to increase the model's parameters to improve baseline performance. The rationale is a greater ensemble of more complex trees should result in improved predictions.

- **`n_estimators = 100`**: Increasing the number of estimators in the forest aims improve predictions by using bagging to reach more accurate results. 

- **`max_depth = 6`**: A gradual increase in the complexity of the trees helps to improve the modelling of the feature-target relationship.

- **`max_features = 4`**: By limiting the maximum features to 4 for node splitting, the model gains more flexibility to consider feature combinations.
---
#### Random Forest Regression Experiment 2 Results

*Table 4: Performance Metrics*

| Model   | R^2 Score | RMSE     |
|---------|-----------|----------|
| RFR(2)  |  0.683    | 0.592    |


<br><br>

![Actual vs Predicted](img/rfr/Model_2-(ActualvsPredicted).png)

*Figure 5: Actual vs Predicted House Prices*
<br><br>

![Residual plot](img\rfr\Model_2-(Residual-plot).png)

*Figure 6: Residual plot*
<br><br>

![Feature importance](img\rfr\Model_2-(Feature-Importance).png)

*Figure 7: Feature importance*
<br><br>

---
#### Random Forest Regression Experiment 2 Discussion

- The model obtained an R-squared score of 0.683 and RMSE of 0.592 (**Table 4**), representing a substantial increase in performance, attributed to increased tree complexity and the ensemble of trees.

- **Figure 5** portrays a closer alignment between actual and predicted values, indicating improved prediction accuracy.

- **Figure 6** displays the model tending towards a more random scatter around the centre line, with the prescence of outliers hindering the predictive capbility.

- **Figure 7** shows median income to be the most influential featur on hosuing prices.

- The increments within parameters led to improved modeling and predictions of housing prices, highlighting the benefits of a greater ensemble of trees and complexity.

---

### Random Forest Regression Experiment 3

#### Random Forest Regression Experiment 3 Description

*Table 5: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(3)  | 200          | 9         | 6            |

Based on the previous experiment, an increase in parameters resulted in a better predictive performance. Therefore, it seems appropriate  to follow the same procedure and assess the model's performance.

- **`n_estimators = 200`**: Due to bagging, I anticipate the predictive performance to increase as a greater ensemble of trees should lead to more accurate predictions.

- **`max_depth = 9`**: Increasing the tree depth within the forest will hopefully faciliate better modelling between the features and housing prices.

- **`max_features = 6`**: Enabling the model to have 6 features to split the nodes, will result in more complex rules to achieve a better decision process.

---
#### Random Forest Regression Experiment 3 Results

*Table 6: Performance Metrics*

| Model   | R^2 Score | RMSE     |
|---------|-----------|----------|
| RFR(3)  |  0.760    | 0.514    |


<br><br>

![Actual vs Predicted](img/rfr/Model_3-(ActualvsPredicted).png)

*Figure 8: Actual vs Predicted House Prices*
<br><br>

![Residual plot](img\rfr\Model_3-(Residual-plot).png)

*Figure 9: Residual plot*
<br><br>

![Feature importance](img\rfr\Model_3-(Feature-Importance).png)

*Figure 10: Feature importance*
<br><br>

---
#### Random Forest Regression Experiment 3 Discussion
- The model achieved an R-squared score of 0.760 and RMSE of 0.514 (**Table 6**), reflecting the importance of a large tree ensemble and greater tree complexity.

- **Figure 8** showcases greater alignment between actual and predicted values, attributed to data points closely clustered around the line.

- **Figure 9** demonstrates a tendency towards a random scatter of residuals around zero, indicating consistency in the model's predictions.

- Consistent with prior experiments, median income emerges as the most influential feature on house prices (**Figure 10**).

- Overall, the notable improvement in model performance is attributed to an increase in estimators in and tree complexity in the forest. To fine-tune the regressor for optimal performance, a grid search is recommended to identify the most suitable parameters for accurate predictions.
---
### Random Forest Regression Experiment 4

#### Random Forest Regression Experiment 4 Description

*Table 7: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(4)  | 500          | 21        | 2            |

This final experiment seeks to determine optimal parameters for the Random Forest Regressor to achieve the best R^2, RMSE scores. To validate previous experiments, the grid search will provide supporting evidence to whether a greater ensemble of trees and complexity results in a better model.

- **`n_estimators = 500`**: 500 estimators in the forest provided the optimal performance for the regressor, yielding the best predictive capability among the tested parameters.

- **`max_depth = 21`**: The optimal tree depth indicates greater tree complexity enhances predictive capabilities.

- **`max_features = 2`**: Interestingly, the optimal number of features required to split a node was found to be two, suggesting that a combination of several features may not be the most effective in achieving accurate predictions.

---
#### Random Forest Regression Experiment 4 Results

*Table 8: Performance Metrics*

| Model   | R^2 Score | RMSE     |
|---------|-----------|----------|
| RFR(4)  |  0.804    | 0.465    |


<br><br>

![Actual vs Predicted](img/rfr/Model_4-(ActualvsPredicted).png)

*Figure 11: Actual vs Predicted House Prices*
<br><br>


![Residual plot](img\rfr\Model_4-(Residualplot).png)

*Figure 12: Residual plot*
<br><br>

![Feature importance](img\rfr\Model_4-(Feature-Importance).png)

*Figure 13: Feature importance*
<br><br>

---
#### Random Forest Regression Experiment 4 Discussion
Utilising the grid search to identify optimal parameters, the model achieved an R-squared score of 0.804 and RMSE of 0.465 (**Table 8**). The results indicate that increasing the number of estimators and tree complexity leads to an improved predictive capability. However, an increase to the maximum features to split a node showed that smaller values yielded better-performing models.

- **Figure 11** reveals a close alignment between actual and predicted values, demonstrating strong predictive capability.

- **Figure 12** shows consistent predictions with a random scatter around zero, but extreme outliers affecting the overall model performance.

- Notably, the most important features are median income, location, and average occupancy, suggesting a deeper understanding of factors influencing housing prices in this final model (**Figure 13**).

- The grid search yielded the optimal parameter combination for a robust predictive model. The experiments indicate that greater complexity and a larger ensemble of trees generally improve model performance, albeit with increased computational demands.
---

### Model comparison & evaluation

![Number_of_estimators](img\rfr\Model_performance-(Number_of_estimators).png)

*Figure 14: Model performance based on the number of estimators*
<br><br>


![Max_depth](img\rfr\Model_performance-(Max_tree_depth).png)

*Figure 15: Model performance based on the maximum tree depth*
<br><br>

![Model Comparison](img\rfr\Model_performance-Comparison.png)

*Figure 16: Model Comparison*
<br><br>

![Learning curve](img\rfr\Learning_curve.png)

*Figure 17: Learning curve*
<br><br>

- **Figures 14 and 15** illustrate the performance improvements with increasing parameters, supported by a model comparison in **Figure 16** showcasing changes across all models.

- **Figure 17** presents a learning curve analysing training data's impact on the validation set, assessing potential overfitting. Initially, the model shows signs of overfitting, but as training samples increase, validation performance improves.
---
## Unsupervised ML

### K-means Clustering

[K-means clustering Sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)

### K-means Experiments

**Parameters used:**

- **`n_clusters`**: This parameter determines the number of clusters to perform clustering with.

---
### K-means Experiment 1

#### K-means Experiment 1 Description

![Optimal clusters](img\clustering\Optimal_clusters.png)
*Figure 18: Inertia and silhoutte plot to find optimal clusters*
<br><br>

The aim of this experiment is to utilise the most optimal clusters that are identified through **Figure 18** experiment seeks to determine optimal parameters for the Random Forest Regressor to achieve the best R^2, RMSE scores. To validate previous experiments, the grid search will provide supporting evidence to whether a greater ensemble of trees and complexity results in a better model.

- **`n_clusters = 3`**: 
#### K-means Experiment 1 Results

![Avg silhoutte plot](img\clustering\3_clusters_silhoutte.png)
*Figure 19: Average silhoutte per cluster*
<br><br>

![Geographical clusters](img\clustering\3_cluster_map.png)
*Figure 20: Geographical clusters based on Median house value*
<br><br>

![Actual clusters](img\clustering\Actual_Clusters.png)

*Figure 21: Actual geographical Median house value*
<br><br>


#### K-means Experiment 1 Discussion


#### K-means Experiment 2 Description


#### K-means Experiment 2 Results

![Avg silhoutte plot](img\clustering\4_cluster_silhoutte.png)
*Figure 22: Average silhoutte per cluster*
<br><br>

![Geographical clusters](img\clustering\4_cluster_map.png)
*Figure 23: Geographical clusters based on Median house value*
<br><br>

![Actual clusters](img\clustering\Actual_Clusters.png)

*Figure 24: Actual geographical Median house value*
<br><br>

#### K-means Experiment 2 Discussion




