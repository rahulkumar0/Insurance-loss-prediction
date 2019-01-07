
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from featexp import get_univariate_plots
from featexp import get_trend_stats
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

#reading dataset
df = pd.read_csv('insurance.csv')
df.head(5)
df.info()
df.describe()

#check null values
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data.head(7)

#scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
df[['age','bmi','children']] = sc.fit_transform(df[['age','bmi','children']])

#dividing the dataset to train and test for analyzing
test = df.iloc[:len(df)//5,:]
train = df.iloc[len(df)//5:,:]

#dividing the data into 10 bins and analyzing the trend and corelation of all numeric features
get_univariate_plots(data=train, target_col='loss_amt')
stats = get_trend_stats(data=train, target_col='loss_amt', data_test=test)

#encoding categorical variable
df = pd.get_dummies(df)
df = df.drop(['sex_female','smoker_no'],axis =1)#removing dummy variable trap

X = df.iloc[:,df.columns!='loss_amt']
y = df.loc[:,['loss_amt']]

#scale data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Applying Grid Search to find the best model and the best parameters
from xgboost import XGBRegressor
xgb_model = XGBRegressor(learning_rate =0.1, n_estimators=1000, max_depth=5,
   min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8, nthread=6, scale_pos_weight=1, seed=27)


from sklearn.model_selection import GridSearchCV
parameters = {
     'colsample_bytree':[0.4,0.6,0.8],
     'min_child_weight':[1.5,6,10],
     'learning_rate':[0.1,0.2,0.07],
     'max_depth':[3,5,7],
     'n_estimators':[50,100,200],
}
grid_search = GridSearchCV(estimator = xgb_model,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           n_jobs = -1)
grid_search = grid_search.fit(X_train, y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#applying xgboost
xgb_model = XGBRegressor(learning_rate =0.1, n_estimators=50, max_depth=3,
                             min_child_weight=4,colsample_bytree=0.8)
xgb_model.fit(X_train,y_train)
y_pred = xgb_model.predict(X_test)

#ann
def model_fun():
    model = Sequential()
    model.add(Dense(9, input_dim=9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(9, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

estimator = KerasRegressor(build_fn=model_fun, nb_epoch=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10, random_state=0)
results = cross_val_score(estimator, X_train, y_train, cv=kfold, n_jobs=1)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))
y_pred_ann = estimator.predict(X_test)

#Learning Curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
plot_learning_curve(xgb_model, 'curve', X, y, (0.7, 1.01), cv=cv, n_jobs=-1)
plt.show()
