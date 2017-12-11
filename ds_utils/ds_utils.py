import pandas as pd
import numpy as np
import sklearn.linear_model as lm
import sklearn.model_selection as ms
import sklearn.ensemble as en
from sklearn.metrics import mean_squared_error
import dill
try:
    import cPickle as pickle
except:
    import pickle

"""
TODO:
    -Build classifier and cluster classes.
    -Implement grid search
    -Pickle
"""


class Predictor(object):
    """Process data, train models, and predict flight delays"""

    def __init__(self, data=None, data_to_predict=None, target=None):
        """Reads in data and initializes some attributes for later

        Args:
            data: preloaded dataframe, default is None
        """
        self.data = data
        self.target_name = target
        self.target = None
        self.model_dict = {'LinearRegression': lm.LinearRegression(),
                           'Lasso': lm.Lasso(),
                           'Ridge': lm.Ridge,
                           'RandomForestRegressor': en.RandomForestRegressor(),
                           'AdaBoostRegressor': en.AdaBoostRegressor(),
                           'GradientBoost': en.GradientBoostingRegressor(),
                           'BaggingRegressor': en.BaggingRegressor(),
                           'RandomForestClassifier': en.RandomForestClassifier()}
        self.features_ = []
        self.selected_features_ = []
        self.model = None
        self.cv_score_ = {}
        self.train = None
        self.test = None
        self.data_to_predict = data_to_predict
        self.predictions = None
        self.train_score_ = None
        self.test_score_ = None
        self.best_params_ = None
        # self.fill_models = {}
        # self.fill_features = {}

    def load_data(self, filepath, sep=","):
        """loads csv or json data from file

        Args:
            filepath (str): full or relative path to file being loaded
            sep (str): delimiter for csv file being loaded, default is ","
        """
        if filepath.split('.')[-1] == 'csv':
            self.data = pd.read_csv(filepath, sep=sep)
        elif filepath.split('.')[-1] == 'json':
            self.data = pd.read_json(filepath)
        else:
            print 'Please select a csv or json file'

    def load_data_to_predict(self, filepath, sep=","):
        """loads csv or json data from file

        Args:
            filepath (str): full or relative path to file being loaded
            sep (str): delimiter for csv file being loaded, default is ","
        """
        if filepath.split('.')[-1] == 'csv':
            self.predict_data = pd.read_csv(filepath, sep=sep)
        elif filepath.split('.')[-1] == 'json':
            self.predict_data = pd.read_json(filepath)
        else:
            print 'Please select a csv or json file'

    def split(self, test_size=0.25, random_state=None):
        self.train, self.test = ms.train_test_split(
            self.data, test_size=test_size, random_state=random_state)
        self.target = self.train[self.target_name]

    def fit(self, model_name, **model_params):
        """Train model on training data

        Args:
            model_name (str): options are
                                {'LinearRegression': lm.LinearRegression(),
                                'Lasso': lm.Lasso(),
                                'Ridge': lm.Ridge,
                                'RandomRorestRegressor': en.RandomForestRegressor(),
                                'AdaBoostRegressor': en.AdaBoostRegressor(),
                                'GradientBoostingRegressor': en.GradientBoostingRegressor(),
                                'BaggingRegressor': en.BaggingRegressor()}
            target (str): column name of target
            features (list): list of column names to use in fit
            **model_params (dict): Parameters to be passed to model

        Returns:
            trained model
        """
        if str(self.target) == 'None':
            self.target = self.data[self.target_name]
        self.data = self.data.loc[:, self.selected_features_]
        model = self.model_dict[model_name]
        model.set_params(**model_params)
        self.model = model.fit(self.data, self.target)

    def grid_search(self, model_name, params):
        """Grid search hyperparameters for regression

        Args:
            model_name (str): options are
                                {'LinearRegression': lm.LinearRegression(),
                                'Lasso': lm.Lasso(),
                                'Ridge': lm.Ridge,
                                'RandomRorestRegressor': en.RandomForestRegressor(),
                                'AdaBoostRegressor': en.AdaBoostRegressor(),
                                'GradientBoostingRegressor': en.GradientBoostingRegressor(),
                                'BaggingRegressor': en.BaggingRegressor()}
            **params (dict): grid of parameters to search over.  Keys are parameters as strings and values are lists of parameter values.
        """
        model = self.model_dict[model_name]
        regr = ms.GridSearchCV(
            estimator=model, param_grid=params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=4)
        regr.fit(self.train[self.selected_features_], self.target)
        self.best_params_ = regr.best_params_
        self.train_score_ = regr.best_score_

    def mean_baseline(self):
        train_mean = np.mean(self.target)
        rmse = np.sqrt(
            np.mean(np.square(self.test[self.target_name] - train_mean)))
        print 'mean baseline RMSE:  {}'.format(rmse)

    def fitCV(self, model_name, n_splits, **model_params):
        """cross-validate model on training data and store score in self.cv_score_

        Note:
            Data must be preprocessed

        Args:
            n_splits (int): number of splits in cross-validation
            model_name (str): options are
                                {'LinearRegression': lm.LinearRegression(),
                                'Lasso': lm.Lasso(),
                                'Ridge': lm.Ridge,
                                'RandomRorestRegressor': en.RandomForestRegressor(),
                                'AdaBoostRegressor': en.AdaBoostRegressor(),
                                'GradientBoostingRegressor': en.GradientBoostingRegressor(),
                                'BaggingRegressor': en.BaggingRegressor()}
            target (str): column name of target
            features (list): list of column names to use in fit
            **model_params (dict): Parameters to be passed to model
        """
        model = self.model_dict[model_name]
        model.set_params(**model_params)
        errors = []
        X = self.train[self.selected_features_]
        y = self.train[self.target_name]
        kf = ms.KFold(n_splits=n_splits, shuffle=True)
        error = []
        for train_index, test_index in kf.split(y):
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            error.append(np.sqrt(skm.mean_squared_error(y_test, predictions)))
        self.cv_score_[model_name] = np.mean(error)

    def predict_missing_values(self, data, model_name, targets, features, **model_params):
        """Predicts missing values based on filled values

        Args:
            model_name: options are {'LinearRegression':
                                lm.LinearRegression(),
                               'Lasso': lm.Lasso(),
                               'Ridge': lm.Ridge,
                               'RandomForestRegressor': en.RandomForestRegressor(),
                               'AdaBoostRegressor': en.AdaBoostRegressor(),
                               'GradientBoost': en.GradientBoostingRegressor(),
                               'BaggingRegressor': en.BaggingRegressor(),
                               'RandomForestClassifier': en.RandomForestClassifier()}
            targets: list of column names with missing values to be filled in
            features: features used to predict missing values
        """
        for target in targets:
            model = self.model_dict[model_name]
            model.set_params(**model_params)
            cols = features + [target]
            train_df = data.loc[:, cols].dropna()
            fill_mask = pd.isnull(data[target])
            model.fit(train_df[features], train_df[target])
            data.loc[fill_mask, target] = model.predict(
                data.loc[fill_mask, features])
            if str(self.test) != 'None':
                print pd.isnull(self.test[target]).any()
                if pd.isnull(self.test[target]).any():
                    fill_mask = pd.isnull(self.test[target])
                    print self.test.loc[fill_mask, features]
                    self.test.loc[fill_mask, target] = model.predict(
                        self.test.loc[fill_mask, features])
        return data
    #
    # def predict_test_missing_values(self):
    #     for target in self.fill_models.keys():
    #         print "filling: {}".format(target)
    #         model = self.fill_models[target]
    #         fill_mask = pd.isnull(self.test[target])
    #         self.test.loc[fill_mask, target] = model.predict(
    #             self.test.loc[fill_mask, self.fill_features[target]])

    def set_features(self, features):
        """Set features to build model with

        Args:
            features (list): list of feature column names
        """
        self.features_ = features

    def select_features(self):
        """trains a Lasso regression and drops features with 0 coefficients"""
        print 'tuning alpha'
        hyper_params_model = lm.LassoCV(normalize=True, n_jobs=-1).fit(
            self.train[self.features_], self.target)
        alpha = hyper_params_model.alpha_
        print 'alpha is: {}'.format(alpha)
        print 'fitting lasso to get coefficients'
        model = lm.Lasso(alpha=alpha, normalize=True)
        model.fit(self.train[self.features_], self.target)
        with open('lasso_coefficients.txt', 'w') as f:
            for coef, feature in sorted(zip(model.coef_, self.features_)):
                f.write('{} : {}\n'.format(feature, coef))
                if coef not in [-0.0, 0.0]:
                    self.selected_features_.append(feature)

    def score(self, model_name, **params):
        model = self.model_dict[model_name]
        model.set_params(**params)
        model.fit(self.train[self.selected_features_], self.target)
        predictions = model.predict(self.test[self.selected_features_])
        self.test_score_ = np.sqrt(mean_squared_error(
            predictions, self.test[self.target_name]))

    def processing(self):
        pass

    def predict(self):
        self.predictions = self.model.predict(self.data_to_predict)

    def dummify(self, data, dummy_columns):
        """Dummifies categorical features

        Args:
            dummy_columns (list): columns to create dummy columns from
        """
        for column in dummy_columns:
            dummies = pd.get_dummies(data[column], prefix=column)
            data = pd.concat((data, dummies), axis=1)
            data.drop(column, axis=1, inplace=True)
        return data

    def pickle_model(self, filename):
        with open(filename, 'wb') as pkl:
            pkl.dump(self.model)


class Classifier(object):

    def __init__(self):
        pass


class Cluster(object):

    def __init__(self):
        pass


if __name__ == '__main__':
    """load data
    process data
    dummify
    set features
    select features
    fit model
    predict
    """
    pass
