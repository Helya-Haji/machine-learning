import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn import tree

X, L = load_wine(return_X_y=True)

kf = KFold(n_splits=10, shuffle=True, random_state=42)  # optional for randomness

ACC_Whole = []

for train_index, test_index in kf.split(X, L):
    X_train, X_test = X[train_index], X[test_index]
    L_train, L_test = L[train_index], L[test_index]

    sc = StandardScaler()
    model = tree.DecisionTreeClassifier()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    model.fit(X_train, L_train)
    ACC = model.score(X_test, L_test)
    ACC_Whole.append(ACC)

ACC_Whole = np.asarray(ACC_Whole)
Mean_ACC = np.mean(ACC_Whole)

print(f"Cross-validated Accuracy: {Mean_ACC:.4f}")
