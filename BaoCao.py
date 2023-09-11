import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv("telecom_churn.csv")
# X = data.iloc[:,1:]
X = data.iloc[:,[1, 4, 5, 6, 7, 8, 9, 10]]
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
  print("Do chinh xac tong the lap" , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Gini, Holdout, dùng baging: " + k)



# ================================================

# ====================KNN=========================
print("====================KNN=========================")

# hold out, dùng baging
i=0
YY = []
for k in range(N):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=k,
    test_size=1/3.0
  )

  Z = KNeighborsClassifier()
  
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
  print("Do chinh xac tong the lap" , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, dùng baging: " + k)

# ===============================================

# ===================Bayes=======================
print("===================Bayes=======================")

# hold out, baging
i=0
YY = []
for k in range(N):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=k,
    test_size=1/3.0
  )

  # Z = GaussianNB()
  Z = GaussianNB()

  ZZ = BaggingClassifier(
    base_estimator=Z,
    n_estimators=10,
    random_state=42)

  ZZ.fit(X_train, y_train)
  yy = ZZ.predict(X_test)
  YY.append(accuracy_score(y_test, yy))
  i = i+1
  acc = accuracy_score(y_test, yy)*100
  print("Do chinh xac tong the lap" , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, baging: " + k)

#=====================



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
  print("Do chinh xac tong the lap" , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Gini, Holdout, dùng baging: " + k)



# ================================================

# ====================KNN=========================
print("====================KNN=========================")

# hold out, dùng baging
i=0
YY = []
for k in range(N):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=k,
    test_size=1/3.0
  )

  Z = KNeighborsClassifier()
  

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))
  i = i+1
  acc = accuracy_score(y_test, yy)*100
  print("Do chinh xac tong the lap" , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, KNeighborsClassifier: " + k)

# ===============================================

# ===================dECISION TREE=======================

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

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))
  i = i+1
  acc = accuracy_score(y_test, yy)*100
  print("Do chinh xac tong the lap" , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Gini, Holdout, DecisionTreeClassifier: " + k)

#Bayes

print("========================Bayes=======================")

# hold out, baging
i=0
YY = []
for k in range(N):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=k,
    test_size=1/3.0
  )

  # Z = GaussianNB()
  Z = GaussianNB()

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))
  i = i+1
  acc = accuracy_score(y_test, yy)*100
  print("Do chinh xac tong the lap" , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, bayes: " + k)