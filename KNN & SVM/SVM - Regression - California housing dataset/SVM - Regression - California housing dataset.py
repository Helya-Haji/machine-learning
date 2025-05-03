import numpy as np
import sklearn as sk
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

data = fetch_california_housing()
X = data.data
T = data.target

X_train, X_test, T_train, T_test = train_test_split(X,T,test_size=0.3)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

Model = SVR()

Model.fit(X_train,T_train)

Estimated_Target_Tests = Model.predict(X_test)

MSE_Test = mean_squared_error(T_test,Estimated_Target_Tests)

print(f"Mean Squared Error on Test Set: {MSE_Test:.2f}")