import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Creat a dataset
df = pd.read_csv(r"C:\Users\emman\OneDrive\Documents\Machine Learning\Ridge, Lasso Regression\Algerian_forest_fires_dataset_cleaned.csv")

# Drop unwanted columns
df = df.drop(['day', 'month', 'year'], axis=1)

df['Classes'] = np.where(df['Classes'].str.contains('not fire'), 0, 1)

# Split the dataset into features (X) and target variable (y)
X = df.drop('FWI', axis=1)
y = df['FWI']

# Train test splits
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=42)

# Feature selection based on correlation
# print(X_train.corr())

# Check for multicollinearity
plt.figure(figsize=(12,10))
corr = X_train.corr()

sns.heatmap(corr, annot=True)
plt.show()

def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr

# Threshold is set by domain expert.

corr_feature = correlation(X_train, 0.85)

X_train.drop(corr_feature, axis=1, inplace=True)

X_test.drop(corr_feature, axis=1, inplace=True)

# Feature Scaling or Standardisation

scaler = StandardScaler()
X_train_Scaled = scaler.fit_transform(X_train)
X_test_Scaled = scaler.transform(X_test)

# Box plot to understand scaling
plt.subplots(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.boxplot(data=X_train)
plt.title('X_train Before Scalling')
sns.boxplot(data=X_train_Scaled)
plt.title('X_test After Scaling')
plt.show()

# region Linear regression model

linreg = LinearRegression()
linreg.fit(X_train_Scaled, y_train)
y_pred = linreg.predict(X_test_Scaled)

# metrics
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test, y_pred)

print('mean_absolute_error:', mae)

plt.scatter(y_test, y_pred)

# endregion

# region Lasso regression

from sklearn.linear_model import Lasso

lassoReg = Lasso()
lassoReg.fit(X_train_Scaled, y_train)
y_pred_lasso = lassoReg.predict(X_test_Scaled)

# metrics
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)
score_lasso = r2_score(y_test, y_pred_lasso)

print('mean_absolute_error_lasso:', mae_lasso)
print('R2 Score lasso', score_lasso)

plt.scatter(y_test, y_pred_lasso)
plt.show()


# endregion

# region Ridge regression

from sklearn.linear_model import Ridge

ridgeReg = Ridge()
ridgeReg.fit(X_train_Scaled, y_train)
y_pred_ridge = ridgeReg.predict(X_test_Scaled)

# metrics
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)
score_ridge = r2_score(y_test, y_pred_ridge)

print('mean_absolute_error_ridge:', mae_ridge)
print('R2 Score ridge', score_ridge)

plt.scatter(y_test, y_pred_ridge)
plt.show()

# endregion

# region Elasticnet Regression

from sklearn.linear_model import ElasticNet

elasticReg = ElasticNet()
elasticReg.fit(X_train_Scaled, y_train)
y_pred_elastic = elasticReg.predict(X_test_Scaled)

# metrics

mae_elastic = mean_absolute_error(y_test, y_pred_elastic)
score_elastic = r2_score(y_test, y_pred_elastic)

print('mean_absolute_error_elastic:', mae_elastic)
print('R2 Score elastic', score_elastic)

plt.scatter(y_test, y_pred_elastic)

plt.show()

# endregion

# Cross Validation

# region Lasso crossvalidation

from sklearn.linear_model import LassoCV

lassoRegCV = LassoCV(cv=5)
lassoRegCV.fit(X_train_Scaled, y_train)

y_pred_lasso_cv = lassoRegCV.predict(X_test_Scaled)
plt.scatter(y_test, y_pred_lasso_cv)
plt.show()

# endregion

# region Ridge regression crossvalidation

from sklearn.linear_model import RidgeCV

ridgeRegCV = RidgeCV(cv=5)
ridgeRegCV.fit(X_train_Scaled, y_train)

y_pred_ridge_cv = ridgeRegCV.predict(X_test_Scaled)

plt.scatter(y_test, y_pred_ridge_cv)
plt.show()


# endregion

# region Elasticnet regression crossvalidation

from sklearn.linear_model import ElasticNetCV

elasticRegCV = ElasticNetCV(cv=5)
elasticRegCV.fit(X_train_Scaled, y_train)

y_pred_elastic_cv = elasticRegCV.predict(X_test_Scaled)

plt.scatter(y_test, y_pred_elastic_cv)

plt.show()

# endregion

# region Pickling the ML models
import pickle

pickle.dump(scaler, open(r'C:\Users\emman\OneDrive\Documents\Machine Learning\End to End Projects\scaler.pkl', 'wb'))
pickle.dump(ridgeReg, open(r'C:\Users\emman\OneDrive\Documents\Machine Learning\End to End Projects\ridge.pkl', 'wb'))