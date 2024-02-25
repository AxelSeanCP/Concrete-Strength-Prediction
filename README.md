# Machine Learning Project Report - Axel Sean Cahyono Putra
![image of concrete compressive strength](img/Compressive-Strength-Of_Concrete.jpg)
# Concrete Compressive Strength
## Project Domain
Based on [[1]](#1-kenneth-a-tutu-david-a-odei-philip-baniba-michael-owusu-concrete-quality-issues-in-multistory-building-construction-in-ghana-cases-from-kumasi-metropolis-case-studies-in-construction-materials-vol-17-december-2022-e01425-httpswwwsciencedirectcomsciencearticlepiis2214509522005575) in Ghana, building collapse incidents are frequent and often result in casualties. From 2000 to 2020, around 60 people died and 140 were injured in 20 cases of building collapse. Even in some major cities in Ghana, the phenomenon of building collapse still occurs, believed to be due to the use of low-quality building materials, especially concrete. In Ghana, concrete production for building materials is done directly at the construction site, as purchasing ready-to-use concrete is expensive. The quality of pre-prepared concrete is certainly better because it meets predetermined quality parameters.

Building materials, especially low-quality concrete, are believed to be a source of building collapses. One way to determine the quality of concrete is by testing its compressive strength, also known as **Concrete Compressive Strength**. The compressive strength of concrete depends on the components used in the mixing process. The amount of cement, water, coarse aggregate, fine aggregate, etc., are factors that determine the compressive strength of concrete, but there are other factors such as temperature, air humidity, and exposure that can also affect the strength of concrete. Therefore, the composition and exposure conditions make the prediction process complex. [[2]](#2-gaoyang-liu-bochau-sun-concrete-compressive-strength-prediction-using-an-explainable-boosting-machine-model-case-studies-in-construction-materials-vol-18-july-2023-e01845-httpswwwsciencedirectcomsciencearticlepiis2214509523000244).

Therefore, the use of machine learning models can predict the compressive strength of concrete based on the data of components used in the concrete mix. Since the unit of concrete strength is a constant value, the suitable machine learning model is **Regression**. The regression model can find patterns within the data related to the compressive strength of concrete (output variable) by analyzing the amount of components used in each concrete mix and determining which component mix and quantity can affect the compressive strength of concrete.

The Regression model can be applied in several stages: Data Collection, Data Preparation, Data Splitting, Model Training, Model Evaluation. After completing these stages, the model with the best evaluation metric will be selected as the model for prediction.

## Business Understanding
As previously explained, sometimes concrete mixing is done on-site with minimal equipment to test the concrete. The use of a regression machine learning model can help predict the compressive strength of concrete before it is used in the construction process. This way, the construction process can reduce excessive spending and also ensure the safety of the building being constructed.

### Problem Statement
- How to predict the compressive strength of concrete based on the data of components used in a concrete mix to increase efficiency and safety
- How to obtain a machine learning model with an error rate below 5%

### Goals
- Successfully predict the compressive strength of concrete using a machine learning model
- Successfully obtain a model with an error rate of less than 5%

### Solution Statement
- Use EDA to understand the nature of the data and identify features that affect Concrete Compressive Strength
- Use multiple machine learning models to predict the compressive strength of concrete based on the given component data. The model to be used is the regression model. Then choose the model with the smallest error, some of the models to be used are:
    1. K-Nearest Neighbors
    2. Support Vector Regressor
    3. Random Forest
    4. XGBoost Regressor
- Choose the model with the lowest error to be used in predicting the compressive strength of concrete

## Data Understanding
The Concrete Compressive Strength dataset can be downloaded through [this link](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength). The dataset contains 9 variables with 8 features and 1 target. Description of variables:
> The features in this dataset are components used in concrete mixtures
- Cement (component 1): amount of cement in the concrete mixture measured in kg in a cubic meter
- Blast Furnace Slag (component 2): amount of slag from blast furnaces in the concrete mixture measured in kg in a cubic meter
- Fly Ash (component 3): amount of fly ash in the concrete mixture measured in kg in a cubic meter
- Water (component 4): amount of water in the concrete mixture measured in kg in a cubic meter
- Superplasticizer (component 5): amount of superplasticizer in the concrete mixture measured in kg in a cubic meter
- Coarse Aggregate (component 6): amount of coarse aggregate in the concrete mixture measured in kg in a cubic meter
- Fine Aggregate (component 7): amount of fine aggregate in the concrete mixture measured in kg in a cubic meter
- Age: age of the concrete in days (1-365)
- Concrete compressive strength: compressive strength of concrete measured in MPa (megapascals), is the target variable in the dataset

### Data Information
The dataset contains 1030 samples  

| idx | column                        | non-null count | dtype   |
|-----|-------------------------------|----------------|---------|
| 0   | Cement                        | 1030 non-null  | float64 |
| 1   | Blast Furnace Slag            | 1030 non-null  | float64 |
| 2   | Fly Ash                       | 1030 non-null  | float64 |
| 3   | Water                         | 1030 non-null  | float64 |
| 4   | Superplasticizer              | 1030 non-null  | float64 |
| 5   | Coarse Aggregate              | 1030 non-null  | float64 |
| 6   | Fine Aggregate                | 1030 non-null  | float64 |
| 7   | Age                           | 1030 non-null  | int64   |
| 8   | Concrete Compressive Strength | 1030 non-null  | float64 |
  

Checking for missing values  

|       |      Cement | Blast Furnace Slag |     Fly Ash |       Water | Superplasticizer | Coarse Aggregate | Fine Aggregate |         Age | Concrete compressive strength |
|------:|------------:|-------------------:|------------:|------------:|-----------------:|-----------------:|---------------:|------------:|------------------------------:|
| count | 1030.000000 | 1030.000000        | 1030.000000 | 1030.000000 | 1030.000000      | 1030.000000      | 1030.000000    | 1030.000000 | 1030.000000                   |
|  mean | 281.167864  | 73.895825          | 54.188350   | 181.567282  | 6.204660         | 972.918932       | 773.580485     | 45.662136   | 35.817961                     |
|  std  | 104.506364  | 86.279342          | 63.997004   | 21.354219   | 5.973841         | 77.753954        | 80.175980      | 63.169912   | 16.705742                     |
|  min  | 102.000000  | 0.000000           | 0.000000    | 121.800000  | 0.000000         | 801.000000       | 594.000000     | 1.000000    | 2.330000                      |
|  25%  | 192.375000  | 0.000000           | 0.000000    | 164.900000  | 0.000000         | 932.000000       | 730.950000     | 7.000000    | 23.710000                     |
|  50%  | 272.900000  | 22.000000          | 0.000000    | 185.000000  | 6.400000         | 968.000000       | 779.500000     | 28.000000   | 34.445000                     |
|  75%  | 350.000000  | 142.950000         | 118.300000  | 192.000000  | 10.200000        | 1029.400000      | 824.000000     | 56.000000   | 46.135000                     |
|  max  | 540.000000  | 359.400000         | 200.100000  | 247.000000  | 32.200000        | 1145.000000      | 992.600000     | 365.000000  | 82.600000                     |

In some components such as *Blast Furnace Slag*, *Fly Ash*, *Superplasticizer*, the min value (minimum value in that column) contains a value of 0.
Usually, the value 0 will be considered as *missing values*, however in this project **the value 0 will be assumed that the component was not used in the mixing process**

### Data Visualization
- Univariate Analysis

![univar_analysis](img/univariate_analysis.png)
Figure 1. Univariate Analysis

It shows the amount of each component used in the mixing process. It can be seen that there are many values of 0 in the *Blast Furnace Slag*, *Fly Ash*, and *Superplasticizer* components. It can be concluded that these three components are rarely used in the concrete mixing process.

- Multivariate Analysis

![corr_matrix](img/correlation_matrix.png)
Figure 2. Multivariate Analysis

Look at the bottom row of the correlation matrix, components such as cement and age have a high correlation with concrete compressive strength, while components such as water do not have much influence on concrete compressive strength. However, the *Water* feature is not dropped because it is believed that water is also an important component in the concrete mixing process.

It can also be seen that many features are not correlated with the target variable, but with this, it can be concluded that the dataset is **non-linear**.

## Data Preparation
The steps taken in Data Preparation:

- Handling outliers: outliers are handled using the IQR method, by removing outliers 1.5 IQR below Q1 and 1.5 IQR above Q3.
- Splitting features and target variables, so that the machine can distinguish which variables need to be used in training.
- Splitting the data into train and test with an 80:20 ratio because the dataset is relatively small, used so that the model can evaluate on new data and prevent overfitting.
- Standardization: the process of transforming data so that it has the same scale, namely mean = 0 and variance = 1, this is necessary for machine learning algorithms to perform better when the data has the same scale.
    > It should be noted that in standardizing the dataset with all numerical features, there is a change from a dataframe to an np array. This can be prevented by converting it back to a dataframe as stated in the code.

## Modelling
The steps in the modeling process for each model:
1. Data Collection
2. Data Preparation
3. Data Splitting into training and testing data
4. Model Training
5. Model Evaluation

Algorithms used:
1. **K-Nearest Neighbor (KNN)**: *KNN* is an algorithm for classifying objects based on data that is closest to the object. This algorithm is used because it is easy to use.
    * Pros: Easy to understand and implement, does not require complex learning or training.
    * Cons: Performance is slow for large datasets, sensitive to non-standardized data, and requires choosing the right parameter K.
    * Parameter:
        - n_neighbors: '6', the number of nearest neighbors to be used for predicting the target value.

2. **Support Vector Regressor (SVM)**: *SVM* is an algorithm to find the best hyperplane that separates 2 classes from the feature space. In the case of regression, this algorithm looks for the hyperplane with the largest margin between the data. This algorithm is used because it is suitable for handling non-linear datasets through the "kernel" parameter.
    * Pros: Effective in datasets with many features, can handle non-linear data through kernels, and tends to be more tolerant to overfitting.
    * Cons: Requires tuning the right parameters, such as kernel and C, and is not efficient for very large datasets.
    * Parameter:
        - kernel: 'rbf', used to transform input data into higher dimensions.
        - 'rbf' = kernel function useful for non-linear data (argument of the kernel parameter).

3. **Random Forest**: *RF* is an algorithm that consists of many decision trees generated randomly, then the average of the final results of each tree will be used for prediction. This algorithm is used because of its ability to handle non-linear data and can provide stable results.
    * Pros: Can handle unstructured data and non-standardized features, robust to outliers and noise, and easy to use.
    * Cons: Potential for overfitting on small datasets with highly diverse features, and difficult to interpret.
    * Parameter:
        - n_estimators: '16', the number of decision trees in the random forest, the more trees, the more complex and computationally expensive.
        - max_depth: '8', the maximum depth of each decision tree.
        - random_state: '69', controls randomness in the model, if given a value, it will be randomly consistent.
        - n_jobs: '-1', the number of jobs to be used in parallel for processing. The default value is **None**, if filled with -1 then it will use all available cpu cores.

4. **Extreme Gradient Boosting (XGBRegressor)**: *XGBRegressor* is an implementation of the gradient boosting algorithm that combines several weak learner models and adds a new learner at each iteration with the goal of improving the previous prediction results. This algorithm is used because it can provide good performance in predicting target values.
    * Pros: Usually provides very good performance, tolerant to overfitting, and efficient in computational time.
    * Cons: Requires careful parameter tuning, and may require more computational processing than other models.
    * Parameter:
        - objective: 'reg:squarederror', the objective function for modeling. the goal is squared error regression loss which is suitable for loss function.
        - n_estimators: '16', the number of decision trees in the ensemble model.
        - max_depth: '8', the maximum depth of each decision tree.
        - learning_rate: '0.3', controls how big the learning step taken at each iteration.
        - subsample: '0.5', The fraction of the dataset to be used for training each tree.
        - colsample_bytree: '0.5', The fraction of features to be used in forming each tree.

## Evaluation
The evaluation metric used is the **root_mean_squared_error (RMSE)** loss function, implemented using the mean_squared_error loss function from sklearn and then taking the square root using numpy.sqrt(). The result is the RMSE loss function.

RMSE or Root Mean Squared Error is a loss function obtained from the process of squaring the error (y_true - y_prediction) and dividing by the count, then taking the square root.

Using this metric, the models can be trained and the error can be measured using the formula:

RMSE = $\displaystyle \left(\frac{\sum (y_i - \hat{y}_i)}{n}\right)^{1/2}$

Where:  
RMSE = root mean square error value  
y  = actual value  
ŷ  = predicted value  
i  = data index  
n  = number of data  

The table below shows the loss for each model:

| Model | Train RMSE | Test RMSE |
|-------|------------|-----------|
| KNN   | 6.356479   | 7.825455  |
| SVR   | 8.290605   | 8.360327  |
| RF    | 3.329867   | 5.268821  |
| XGR   | 2.857279   | 4.971411  |

The plot below shows the loss for each model:

![model loss](img/model_loss_plot.png)  
Figure 3. Model Loss Plot

It can be seen that the XGBRegressor model has the lowest loss among the four models. Thus, it is the best model among the others.

The table below shows the prediction results for each model:

|  id | y_true | KNN_prediction | SVR_prediction | RF_prediction | XGR_prediction |
|----:|-------:|---------------:|---------------:|--------------:|---------------:|
| 418 | 11.98  | 18.5           | 17.6           | 14.1          | 14.500000      |
| 777 | 31.84  | 32.0           | 28.1           | 35.0          | 32.099998      |
| 176 | 56.50  | 60.8           | 45.8           | 58.5          | 58.099998      |
| 481 | 61.07  | 52.4           | 57.1           | 54.9          | 60.500000      |
| 435 | 41.20  | 35.6           | 31.0           | 40.2          | 36.500000      |
|  53 | 49.19  | 48.4           | 40.8           | 48.1          | 53.900002      |
| 904 | 23.79  | 34.6           | 29.8           | 25.6          | 33.700001      |
| 266 | 38.50  | 27.3           | 29.9           | 33.5          | 32.099998      |
| 999 | 15.57  | 21.6           | 27.7           | 15.8          | 16.900000      |
| 380 | 57.23  | 61.7           | 43.3           | 60.8          | 66.699997      |

Based on this data, it can be seen that the regression model can predict the compressive strength of concrete based on the given component data. The model with an error less than 5% is the XGBRegressor model. It can also be seen that the prediction results from the XGBRegressor model are close to the actual values, although sometimes slightly off.

This project still needs improvement, especially in the modeling part. For the XGBRegressor model, hyperparameter tuning can be done to obtain a smaller error in order to predict the compressive strength of concrete with as small an error as possible.

# Conclusion
In the model evaluation, the evaluation metric results show that the *XGBRegressor* model provides the best results, with an *RMSE* error of 4.97 on the test data. This indicates that the *XGBRegressor* model can predict concrete compressive strength with a low error of less than 5% of the actual value.

The prediction results of other models such as *Random Forest* also provide relatively small errors compared to the *KNN* and *Support Vector Regressor* models, which have slightly larger errors than the others. Nevertheless, all models are capable of providing good predictions for concrete compressive strength.

This project has successfully achieved its goal of predicting concrete compressive strength using machine learning models. However, hyperparameter tuning is needed for the XGBRegressor model to achieve even lower error values for more reliable predictions.

# References
##### [1] Kenneth A. Tutu, David A. Odei, Philip Baniba, Michael Owusu, "Concrete quality issues in multistory building construction in Ghana: Cases from Kumasi metropolis," *Case Studies in Construction Materials, Vol 17, December 2022, e01425* https://www.sciencedirect.com/science/article/pii/S2214509522005575
##### [2] Gaoyang Liu, Bochau Sun, "Concrete compressive strength prediction using an explainable boosting machine model," *Case Studies in Construction Materials, Vol 18, July 2023, e01845* https://www.sciencedirect.com/science/article/pii/S2214509523000244