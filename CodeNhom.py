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

# ==============Decison Tree===================
print("==============Decison Tree===================")
# Holdout, không baging, không random forest, gini
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=1/3.0
  )

  Z = DecisionTreeClassifier(
    criterion='gini',
    random_state=49,
    max_depth=10,
    min_samples_leaf=5
  )


  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Gini, Hold out, Không baging, không random forest: " + k)

# Holdout, không baging, không random forest, entropy
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=1/3.0
  )

  Z = DecisionTreeClassifier(
    criterion='entropy',
    random_state=49,
    max_depth=10,
    min_samples_leaf=5
  )


  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Entropy, Hold out, Không baging, không random forst: " + k)

# KFold, không baging, không random forest, gini
kf = KFold(n_splits=50)
YY = []
for i, j in kf.split(X):

  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = DecisionTreeClassifier(
    criterion='gini',
    random_state=49,
    max_depth=10,
    min_samples_leaf=5
  )

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Gini, K-fold, Không baging, không random forst: " + k)

# KFold, không baging, không random forest, entropy
kf = KFold(n_splits=50)
YY = []
for i, j in kf.split(X):

  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = DecisionTreeClassifier(
    criterion='entropy',
    random_state=49,
    max_depth=10,
    min_samples_leaf=5
  )

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Entropy, K-fold, Không baging, không random forst: " + k)

# baging, Holdout, gini
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=1/3.0
  )

  Z = DecisionTreeClassifier(
    criterion='gini',
    random_state=49,
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

k = str(round(sum(YY) / len(YY) * 100))
print("Gini, Holdout, dùng baging: " + k)

# baging, Holdout, entropy
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=1/3.0
  )

  Z = DecisionTreeClassifier(
    criterion='entropy',
    random_state=49,
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

k = str(round(sum(YY) / len(YY) * 100))
print("Entropy, Holdout, dùng baging: " + k)

# Random forest, hold out
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=1/3.0
  )

  Z = RandomForestClassifier(
    n_estimators=10,
    random_state=42
  )

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, random forest: " + k)

# baging, KFold, gini
kf = KFold(n_splits=50)

YY = []

for i, j in kf.split(X):

  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = DecisionTreeClassifier(
    criterion='gini',
    random_state=49,
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


k = str(round(sum(YY) / len(YY) * 100, 2))
print("Gini, K-fold, dùng baging: " + k)

# baging, KFold, entropy
kf = KFold(n_splits=50)

YY = []

for i, j in kf.split(X):

  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = DecisionTreeClassifier(
    criterion='entropy',
    random_state=49,
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

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Entropy, K-fold, dùng baging: " + k)

# Random forsest, k fold
YY = []
for i, j in kf.split(X):

  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = RandomForestClassifier(
    n_estimators=10,
    random_state=42
  )

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("K-fold, Random forest: " + k)

# ================================================

# ====================KNN=========================
print("====================KNN=========================")
# Holdout, không dùng baging
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=1/3.0
  )

  Z = KNeighborsClassifier()

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, không dùng baging: " + k)

# hold out, dùng baging
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
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

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, dùng baging: " + k)

# Kfold, không baging
kf = KFold(n_splits=50)

YY = []
for i, j in kf.split(X):
  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = KNeighborsClassifier()

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("kfold, không dùng baging: " + k)

# Kfold, dùng baging
YY = []
for i, j in kf.split(X):
  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = KNeighborsClassifier()

  ZZ = BaggingClassifier(
    base_estimator=Z,
    n_estimators=10,
    random_state=42
  )

  ZZ.fit(X_train, y_train)
  yy = ZZ.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Kflod, dùng baging: " + k)
# ===============================================

# ===================Bayes=======================
print("===================Bayes=======================")
# Holdout, không baging
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
    test_size=1/3.0
  )

  # Z = GaussianNB()
  Z = GaussianNB()

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, không dùng baging là: " + k)

# hold out, baging
YY = []
for k in range(50):
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=42,
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

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, baging: " + k)

# kfold, không baging
kf = KFold(n_splits=50)

YY = []
for i, j in kf.split(X):
  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = GaussianNB()

  Z.fit(X_train, y_train)
  yy = Z.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("kfold, không dùng baging: " + k)

# kfold, dùng baging
YY = []
for i, j in kf.split(X):
  X_train, X_test = X.iloc[i], X.iloc[j]
  y_train, y_test = y.iloc[i], y.iloc[j]

  Z = GaussianNB()
  ZA = BaggingClassifier(
    base_estimator=Z,
    n_estimators=10,
    random_state=42
  )

  ZZ.fit(X_train, y_train)
  yy = ZZ.predict(X_test)
  YY.append(accuracy_score(y_test, yy))

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Kfold, dùng baging là: " + k)