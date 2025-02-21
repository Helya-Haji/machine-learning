def Evaluate(predict,Label):
    N = Label.shape[0]
    H = predict == Label
    Nt = np.sum(H==1)
    Accurcy = Nt/N 
    Loss = 1 - Accurcy 
    return Accurcy, Loss 
    

import numpy as np 
import sklearn as sk 

from sklearn.datasets import load_iris
X,L = load_iris(return_X_y=True) 


from sklearn.model_selection import train_test_split
X_train, X_test, L_train, L_test = train_test_split(X,L,test_size=0.3)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
Model_LDA = LinearDiscriminantAnalysis()

Model_LDA.fit(X_train,L_train)

Predict_LDA = Model_LDA.predict(X_test)

Accurcy_LDA, Loss_LDA = Evaluate(Predict_LDA,L_test)

print("The accuracy of LDA is:",Accurcy_LDA)
print("The loss of LDA is:" ,Loss_LDA)

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
Model_QDA = QuadraticDiscriminantAnalysis()

Model_QDA.fit(X_train,L_train)

Predict_QDA = Model_QDA.predict(X_test)

Accurcy_QDA, Loss_QDA = Evaluate(Predict_QDA,L_test)

print("The accuracy of QDA is:" , Accurcy_QDA)
print("The loss of QDA is:" , Loss_QDA)