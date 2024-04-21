# Report

## Contents

- [Supervised ML](#supervised-ml)
  - [Random Forest Regression](#random-forest-regression)
  - [California Housing Dataset](#california-housing-dataset)
  - [Random Forest Regression Experiments](#random-forest-regression-experiments)
    - [Random Forest Regression Experiment 1](#random-forest-regression-experiment-1)
    - [Random Forest Regression Experiment 1 Description](#random-forest-regression-experiment-1-description)
    - [Random Forest Regression Experiment 1 Results](#random-forest-regression-experiment-1-results)
    - [Random Forest Regression Experiment 1 Discussion](#random-forest-regression-experiment-1-discussion)
- [Unsupervised ML](#unsupervised-ml)


## Supervised ML

### Random Forest Regression

[Random Forest Regression (Overview)](https://builtin.com/data-science/random-forest-algorithm)

In this AI and ML project, the [Random Forest Regressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html) was utilised to predict California housing prices. The Random Forest algorithm is an ensemble method that combines numerous decision trees and aggregates all predictions from the trees to achieve more accurate results (**bagging**). Through combining multiple trees and **bootstrapping**, the model develops an ability to generalise well to new samples whilst mitigating the risk of overfitting to the training data. 

The decision to employ the Random Forest algorithm to predict California housing prices, was due to the model’s capability to handle complex and non-linear relationships between features and the target variable. Features such as location, population, and house age may not exhibit linear relationships with housing prices, making models such as linear regression unsuitable for accurate predictions.

Furthermore, the ensemble of trees enables the handling of extreme outliers or anomalies within the housing market as the model typically can adjust well to unseen samples. Finally, a beneficial characteristic of Random Forest regression is feature importance, which provides valuable insights into the best predictors of house prices.

---

### California Housing Dataset


The  [California Housing dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset) consists of 8 features (visible in the [ML Experiments Notebook](../src/ML_experiments.ipynb)) describing different geographical locations within California and their respective median house values for districts. Each instance in the dataset represents the median house value for a block group, totalling 20,640 instances.

Before implementing Random Forest Regression on the dataset, I conducted initial pre-processing steps to address outliers within the features. I utilised the Interquartile Range (**IQR**) method to manage outliers (**Q1/Q3 +- 1.5 * IQR**). This approach was applied to all features, excluding the median house value and longitude/latitude, as one is the target variable, and the latter are geographical coordinates. Following pre-processing, the dataset retained 16,840 instances.

---
### Random Forest Regression Experiments

**Parameters used:**

- **`n_estimators`**: This parameter determines the number of decision trees to be used in the forest.


- **`max_depth`**: This parameters controls the maximum depth of each decision tree in the Random Forest.


- **`max_features`**: This parameter specifies the maximum number of features Random Forest is allowed to consider when looking for the best split at each node.

These parameters were selected to strike a balance between model performance, computational efficiency, and the complexity of the trees in the Random Forest Regressor.

---
### Random Forest Regression Experiment 1

#### Random Forest Regression Experiment 1 Description

A description of how you experimented with the model (e.g. parameter tuning) and the goal of the experiment. You can refer to other files in your project, but you should include here things like a table containing the configuration you employed, what you changed, and why you changed it. What exactly you include here will depend on the technique you've selected and the experiments you choose to run.

*Table 1: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(1)  | 50           | 3         | 2            |


The goal of this experiment was to evaluate the Random Forest model's performance with a reduced number of trees, limited tree depth, and a small number of selected features. This served as a starting point for assessing the model's behaviour and laying the groundwork for subsequent experiments where parameters and features could be further adjusted and expanded.

- **`n_estimators = 50`**: Starting below the default 100, this parameter represents the number of trees in the Random Forest. By using 50 trees, the experiment aimed to assess model performance with reduced computational intensity, as fewer trees require less computation.

- **`max_depth = 3`**: This parameter determines the maximum depth of each decision tree. A depth of 3 was chosen for efficiency and to provide a clear view of a typical tree's decision process. A lower depth is less computationally intensive and allows for better understanding of the tree's structure.

- **`max_features = 2`**: Initially limited to 2, `max_features` specifies the maximum number of features considered at each node. This choice allowed for observing model performance with simplicity before potentially increasing complexity with more features.

---

#### Random Forest Regression Experiment 1 Results

The results of the experiment. This should include any graphs or tables that you have generated as well as any relevant metrics or statistics that you have calculated. You can include here qualitative observations, comparisons with a baseline/other parameters (if relevant), etc. Consider your presentation of results carefully to show concise but clear information without just bombarding the reader with plots and text!


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

Here, you should include an analysis and evaluation of the results of your experiments. This can include broader considerations than the results, such as: a discussion on the significance and implications of the results with respect to the data and, critically, the model itself; an assessment of the model's performance linked to the experiments; and any observed trends or anomalies; reflection on the experiment's limitations and potential biases; any limitations of the experiment and any potential improvements that could be made; etc.

---

### Random Forest Regression Experiment 2

##### Random Forest Regression Experiment 2 Description

A description of how you experimented with the model (e.g. parameter tuning) and the goal of the experiment. You can refer to other files in your project, but you should include here things like a table containing the configuration you employed, what you changed, and why you changed it. What exactly you include here will depend on the technique you've selected and the experiments you choose to run.

*Table 3: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(2)  | 100           | 6        | 4           |


The goal of this experiment was to evaluate the Random Forest model's performance with a reduced number of trees, limited tree depth, and a small number of selected features. This served as a starting point for assessing the model's behaviour and laying the groundwork for subsequent experiments where parameters and features could be further adjusted and expanded.

- **`n_estimators = 100`**: 

- **`max_depth = 6`**: 

- **`max_features = 4`**: 

---
#### Random Forest Regression Experiment 2 Results

The results of the experiment. This should include any graphs or tables that you have generated as well as any relevant metrics or statistics that you have calculated. You can include here qualitative observations, comparisons with a baseline/other parameters (if relevant), etc. Consider your presentation of results carefully to show concise but clear information without just bombarding the reader with plots and text!


*Table 4: Performance Metrics*

| Model   | R^2 Score | RMSE     |
|---------|-----------|----------|
| RFR(2)  |  0.683     | 0.592   |


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

Here, you should include an analysis and evaluation of the results of your experiments. This can include broader considerations than the results, such as: a discussion on the significance and implications of the results with respect to the data and, critically, the model itself; an assessment of the model's performance linked to the experiments; and any observed trends or anomalies; reflection on the experiment's limitations and potential biases; any limitations of the experiment and any potential improvements that could be made; etc.

---

### Random Forest Regression Experiment 3

##### Random Forest Regression Experiment 3 Description

A description of how you experimented with the model (e.g. parameter tuning) and the goal of the experiment. You can refer to other files in your project, but you should include here things like a table containing the configuration you employed, what you changed, and why you changed it. What exactly you include here will depend on the technique you've selected and the experiments you choose to run.

*Table 5: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(3)  | 200          | 9         | 6            |


The goal of this experiment was to evaluate the Random Forest model's performance with a reduced number of trees, limited tree depth, and a small number of selected features. This served as a starting point for assessing the model's behaviour and laying the groundwork for subsequent experiments where parameters and features could be further adjusted and expanded.

- **`n_estimators = 200`**: 

- **`max_depth = 9`**: 

- **`max_features = 6`**: 

---
#### Random Forest Regression Experiment 3 Results

The results of the experiment. This should include any graphs or tables that you have generated as well as any relevant metrics or statistics that you have calculated. You can include here qualitative observations, comparisons with a baseline/other parameters (if relevant), etc. Consider your presentation of results carefully to show concise but clear information without just bombarding the reader with plots and text!

---

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

Here, you should include an analysis and evaluation of the results of your experiments. This can include broader considerations than the results, such as: a discussion on the significance and implications of the results with respect to the data and, critically, the model itself; an assessment of the model's performance linked to the experiments; and any observed trends or anomalies; reflection on the experiment's limitations and potential biases; any limitations of the experiment and any potential improvements that could be made; etc.


---
## Random Forest Regression Experiment 4

##### Random Forest Regression Experiment 4 Description

A description of how you experimented with the model (e.g. parameter tuning) and the goal of the experiment. You can refer to other files in your project, but you should include here things like a table containing the configuration you employed, what you changed, and why you changed it. What exactly you include here will depend on the technique you've selected and the experiments you choose to run.

*Table 7: Model parameters*

| Model   | n_estimators | max_depth | max_features |
|---------|--------------|-----------|--------------|
| RFR(4)  | 500          | 21        | 2            |


The goal of this experiment was to evaluate the Random Forest model's performance with a reduced number of trees, limited tree depth, and a small number of selected features. This served as a starting point for assessing the model's behaviour and laying the groundwork for subsequent experiments where parameters and features could be further adjusted and expanded.

- **`n_estimators = 500`**: 

- **`max_depth = 21`**: 

- **`max_features = 2`**: 

---
#### Random Forest Regression Experiment 4 Results

The results of the experiment. This should include any graphs or tables that you have generated as well as any relevant metrics or statistics that you have calculated. You can include here qualitative observations, comparisons with a baseline/other parameters (if relevant), etc. Consider your presentation of results carefully to show concise but clear information without just bombarding the reader with plots and text!

---

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

Here, you should include an analysis and evaluation of the results of your experiments. This can include broader considerations than the results, such as: a discussion on the significance and implications of the results with respect to the data and, critically, the model itself; an assessment of the model's performance linked to the experiments; and any observed trends or anomalies; reflection on the experiment's limitations and potential biases; any limitations of the experiment and any potential improvements that could be made; etc.

---



### Model comparsion & evaluation

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
