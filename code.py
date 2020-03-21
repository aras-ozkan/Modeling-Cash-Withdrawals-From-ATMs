import datetime
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

def preprocessT(datasetind, dataset1):
    dataset = np.vstack((dataset1, datasetind))
    id = dataset[:, 0][:, np.newaxis]
    reg = dataset[:, 1][:, np.newaxis]
    day = dataset[:, 2][:, np.newaxis]
    month = dataset[:, 3][:, np.newaxis]
    year = dataset[:, 4][:, np.newaxis]
    trx = dataset[:, 5][:, np.newaxis]
    trx = trx - 1
    year = year - 2018
    id = id / (np.mean(id) - np.min(id))
    ohe = OneHotEncoder(categorical_features=[0])
    enc = ohe.fit(day)
    day = enc.transform(day).toarray()
    enc = ohe.fit(month)
    month = enc.transform(month).toarray()
    enc = ohe.fit(reg)
    reg = enc.transform(reg).toarray()
    data = np.concatenate((reg, day, month, trx, year, id), axis=1)
    data = data[:940]
    return data


def preprocess(dataset):
    id = dataset[:, 0][:, np.newaxis]
    reg = dataset[:, 1][:, np.newaxis]
    day = dataset[:, 2][:, np.newaxis]
    month = dataset[:, 3][:, np.newaxis]
    year = dataset[:, 4][:, np.newaxis]
    trx = dataset[:, 5][:, np.newaxis]
    trx = trx - 1
    year = year - 2018
    id = id/(np.mean(id) - np.min(id))
    ohe = OneHotEncoder(categorical_features=[0])
    enc = ohe.fit(day)
    day = enc.transform(day).toarray()
    enc = ohe.fit(month)
    month = enc.transform(month).toarray()
    enc = ohe.fit(reg)
    reg = enc.transform(reg).toarray()
    data = np.concatenate((reg, day, month, trx, year, id), axis=1)
    return data


def convert_to_learner_preds(x_train):
    learner_preds = []
    tree_pred = tree.predict(x_train)
    mlp_pred = mlp.predict(x_train)
    learner_preds.append(tree_pred)
    learner_preds.append(mlp_pred)
    learner_preds = np.asarray(learner_preds).transpose()
    return learner_preds


def RMSE(preds, true):
    return np.sqrt(((preds - true) ** 2).sum() / preds.shape[0])


if __name__ == '__main__':

    # IMPORT DATA

    df_x = pd.read_csv('training_data.csv')
    X = df_x[['IDENTITY', 'REGION', 'DAY', 'MONTH', 'YEAR', 'TRX_TYPE']].values
    y = df_x['TRX_COUNT'].values
    df_test = pd.read_csv('test_data.csv')
    X_test_real = df_test.values

    # PREPROCESS to X values

    X_processed = preprocess(X)
    X_test_real_preprocessed = preprocessT(X, X_test_real)

    # SPLIT TRAINING and TESTING DATA for ENSEMBLERS and LEARNERS
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.1)
    X_train, X_ens, y_train, y_ens = train_test_split(X_train, y_train, test_size=0.25)


    # LEARNERS Training and Testing with train data

    # Random Forest
    tree = RandomForestRegressor(n_estimators=200, n_jobs=4)
    tree = tree.fit(X_train, y_train.ravel())
    print("Random Forest RMSE before training whole data: ", RMSE(tree.predict(X_test), y_test))

    # Multilayer perceptron
    mlp = MLPRegressor(epsilon=1, max_iter=500, random_state=0)
    mlp = mlp.fit(X_train, y_train.ravel())
    print('Multilayer Perceptron RMSE before training whole data: ', RMSE(mlp.predict(X_test), y_test))

    # ENSEMBLER = Gradient Boosting Regressor

     
    test_preds = GBReg.predict(convert_to_learner_preds(X_test_real_preprocessed))
    test_preds_df = pd.DataFrame(test_preds)
    test_preds_df.to_csv(path_or_buf='testrot.csv')
