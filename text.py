import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("telecom_churn.csv")
X = data.iloc[:,1:]
y = data.iloc[:,0]

N = 10

# ==============Decison Tree===================
print("==============Decison Tree===================")

# baging, Holdout, gini
YY = []
i = 0
for k in range(N):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=k,
    test_size=1/3.0
  )

  Z = DecisionTreeClassifier(
    criterion='gini',
    random_state=42,
    max_depth=10,
    min_samples_leaf=5
  )

  ZZ = BaggingClassifier(
    base_estimator=Z,
    n_estimators=10,
    random_state=42
  )

  ZZ.fit(X_train, y_train)
  yy = ZZ.predict(X_test)
  YY.append(accuracy_score(y_test, yy))
  i = i+1
  acc = accuracy_score(y_test, yy)*100
  print("Do chinh xac tong the lap " , i , ":", acc, "%")
  print(len(yy))

k = str(round(sum(YY) / len(YY) * 100))
print("Gini, Holdout, dùng baging: " + k)
