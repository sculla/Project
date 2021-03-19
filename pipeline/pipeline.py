import tensorflow
import keras
import warnings
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

import numpy as np


class Pipe(object):

    def __init__(self):
        warnings.filterwarnings("ignore")
        self.pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('reduce_dim', PCA()),
            ('regressor', Ridge())
        ])

    def predict(self):
        data = load_boston()
        X_train, X_test, y_train, y_test = train_test_split(data['data'], data['target'])
        n_features_to_test = np.arange(1, 15)
        alpha_to_test = 2.0 ** np.arange(-6, +6)
        scalers_to_test = [StandardScaler(), RobustScaler(), QuantileTransformer()]
        params = [
            {'scaler': scalers_to_test,
             'reduce_dim': [PCA()],
             'reduce_dim__n_components': n_features_to_test,
             'regressor__alpha': alpha_to_test},

            {'scaler': scalers_to_test,
             'reduce_dim': [SelectKBest(f_regression)],
             'reduce_dim__k': n_features_to_test,
             'regressor__alpha': alpha_to_test}
        ]

        pipe = self.pipe.fit(X_train, y_train)
        print('Testing score: ', pipe.score(X_test, y_test))
        gridsearch = GridSearchCV(self.pipe, params, verbose=1).fit(X_train, y_train)
        print('Final score is: ', gridsearch.score(X_test, y_test))
        print(gridsearch.best_params_)


if __name__ == '__main__':
    p = Pipe()
    p.predict()
