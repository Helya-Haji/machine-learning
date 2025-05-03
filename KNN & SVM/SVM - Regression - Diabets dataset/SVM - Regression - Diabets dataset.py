import numpy as np
import sklearn as sk
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error

X,T = load_diabetes(return_X_y=True)

X_train, X_test, T_train, T_test = train_test_split(X,T,test_size=0.3)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

Model = SVR()

Model.fit(X_train,T_train)

Estimated_Target_Tests = Model.predict(X_test)

MSE_Test = mean_squared_error(T_test,Estimated_Target_Tests)

print(f"Mean Squared Error on Test Set: {MSE_Test:.2f}")