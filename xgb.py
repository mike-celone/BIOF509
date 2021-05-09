import pandas as pd #load and manipulate data, one-hot encoding
import numpy as np
import xgboost as xgboost #XGBoost algorithm
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV #for cross validation
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from xgboost import plot_importance
from matplotlib import pyplot

#Import dataset with variables from PanTHERIA and Ecological Traits of the World's Primates
df = pd.read_csv('dfnew.csv')
df.head()

#Setting the species binomial name as the index
df = df.set_index('Species')

#check the number of missing in each column
df.isna().sum()

#Following the Han et al, 2019 paper, we remove sparse features (variables with data missing for more than 80% of species)

#explore the data types
df.dtypes

#Split dataset into dependent (y) and independent (X) variables
X = df.drop('MAYV', axis=1).copy()
X.head()
y = df['MAYV'].copy()
y.head()

#One-hot encoding of catgorical variables
X_encoded = pd.get_dummies(X, columns=['TrophicGuild',
                                       'IUCN',
                                       'Pop_T'])

#change missing values to the sparse value expected by XGBoost which is zero 
X_encoded = X_encoded.fillna(0)

#Check to see how unbalanced the data set is
sum(y)/len(y)

#Train-test split, maintain proportion of 1s and 0s
X_train, X_test, y_train, y_test = train_test_split(X_encoded, 
                                                    y, 
                                                    random_state=1, 
                                                    test_size=0.25, 
                                                    stratify=y)

#Run initial model using AUC to evaluate
xgb = xgboost.XGBClassifier(objective='binary:logistic', 
                            missing=None, 
                            seed=20, 
                            use_label_encoder=False)
xgb.fit(X_train,
        y_train,
        verbose=True,
        early_stopping_rounds=20,
        eval_metric='auc',
        eval_set=[(X_test, y_test)])

plot_confusion_matrix(xgb,
                      X_test,
                      y_test)

#Experiment with a 50/50 train-test split and re-run the model above
X_train, X_test, y_train, y_test = train_test_split(X_encoded, 
                                                    y, 
                                                    random_state=1, 
                                                    test_size=0.5, 
                                                    stratify=y)

#Experiment with some different parameter values.
#For scale_pos_weight parameter XGBoost recommends 
#using (sum 0s)/(sum 1s). Therefore, we use 102/15=7
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.1, 0.01, 0.05],
    'scale_pos_weight': [1, 3, 5, 7, 10]}
 
params=GridSearchCV(
    estimator=xgboost.XGBClassifier(objective='binary: logistic',
                                    seed=1,
                                    subsample=0.9,
                                    colsample_bytree=0.6,
                                    use_label_encoder=False),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=0,
    n_jobs=10,
    cv=5
   )

params.fit(X_train,
           y_train,
           early_stopping_rounds=10,
           eval_metric='auc',
           eval_set=[(X_test, y_test)],
           verbose=False)
print(params.best_params_)

#According to the grid search, the optimal parameters are
#learning_rate=0.1, max_depth=3, scale_pos_weight=1
#We also use colsample_bytree to prevent overfitting
xgb = xgboost.XGBClassifier(seed=1,
                            objective='binary:logistic',
                            max_depth=3,
                            learning_rate=0.1,
                            scale_pos_weight=1,
                            colsample_bytree=0.9,
                            use_label_encoder=False)
xgb.fit(X_train,
        y_train,
        verbose=True,
        early_stopping_rounds=20,
        eval_metric='auc',
        eval_set=[(X_test, y_test)])

plot_confusion_matrix(xgb,
                      X_test,
                      y_test)

#Plot variable importance
plot_importance(xgb)
pyplot.show()

#Calculate probabilities for the species
xgb.fit(X_encoded, y)
pred = xgb.predict_proba(X_encoded)
print(pred)

