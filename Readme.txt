Detailed EDA and Prediciton of diabetes (https://www.kaggle.com/code/jatinpreetsingh/github-diabetes-detailed-eda-and-prediction , <-- for exact notebook)

In this project of analysing and predicting whether or not a patient has Diabetes , we will use RandomForestClassifier with hyperparameter tuning

Robustness to Overfitting: Random Forests have built-in mechanisms to reduce the risk of overfitting, which occurs when a model becomes too complex and performs well on the training data but fails to generalize to new, unseen data. The model achieves this by constructing multiple decision trees using random subsets of the data and features, and then aggregating their predictions. This ensemble approach helps to reduce variance and increase generalization performance.

Handling Nonlinear Relationships: Random Forests are capable of capturing complex nonlinear relationships between features and the target variable. Each decision tree in the ensemble learns from different subsets of data, allowing the model to capture different interaction patterns between features. By combining the predictions of multiple trees, the model can provide a more accurate and robust classification.

Feature Importance Estimation: Random Forests provide a measure of feature importance, indicating the relative contribution of each input variable in making accurate predictions. This information can be valuable for feature selection, identifying the most relevant variables for the classification task, and gaining insights into the underlying data patterns.

Handling High-Dimensional Data: Random Forests can handle high-dimensional data efficiently. They can effectively deal with datasets containing a large number of features without suffering from the "curse of dimensionality" as much as some other models. The random feature selection in each tree helps prevent overfitting and reduces the impact of irrelevant or noisy features.

In this project we could find class imbalances between predictions where 90% of the predictions are negative (0) , and only 10% are positive(1) , this could lead to our model bending towards negative prediction and only learn about that data, and could be failure at predicting positive values. This could be a problem because we could have accuracy over 90% while hiding the fact that it only predicts negative values.

So for this reason we have to use certain Sampling methods such as SMOTE and RandomUnderSampler

*SMOTE (Synthetic Minority Over-sampling Technique) and RandomUnderSampler are two popular techniques used in machine learning to address class imbalance in datasets. *

SMOTE: SMOTE is an oversampling technique that aims to balance the class distribution by creating synthetic samples of the minority class. It works by synthesizing new instances along the line segments connecting similar minority class instances. This helps in increasing the representation of the minority class and can improve the performance of machine learning models when the data is imbalanced.

RandomUnderSampler: RandomUnderSampler is an undersampling technique that aims to balance the class distribution by reducing the number of instances in the majority class. It randomly selects a subset of instances from the majority class, equal to the number of instances in the minority class. This helps in reducing the dominance of the majority class and can be effective when the dataset is extremely imbalanced or when the dataset size is large.
