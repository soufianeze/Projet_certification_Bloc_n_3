import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

def train_and_evaluate(model, X_train, X_test, y_train, y_test, list_of_metrics):
  # entrainement du modele
  model.fit(X_train, y_train)

  # le dictionnaire qui va contenir les scores
  dict_of_scores = {}

  # evaluate model on Train_set on all metrics

  y_pred = model.predict(X_train)
  for metric in list_of_metrics:
    dict_of_scores[f'TRAIN_{metric.__name__}'] = np.round(metric(y_train, y_pred), 3)


  # evaluate model on Test_set on all metrics
  y_pred = model.predict(X_test)
  for metric in list_of_metrics:
    dict_of_scores[f'TEST_{metric.__name__}'] = np.round(metric(y_test, y_pred), 3)


  return pd.Series(dict_of_scores)



def compare_models(list_of_models, X_train, X_test, y_train, y_test, list_of_metrics):
  list_of_scores = []
  for model in list_of_models:
    scores = train_and_evaluate(model, X_train, X_test, y_train, y_test, list_of_metrics)
    scores.name = model.__class__.__name__
    list_of_scores.append(scores)

  df = pd.concat(list_of_scores, axis=1)
  df = df.T
  df = df.sort_values(by='TEST_r2_score', ascending=False)
  return df




