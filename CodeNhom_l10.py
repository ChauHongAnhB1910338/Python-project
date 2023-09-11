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
  X_train, X_test, y_train, y_test = train_test_split( #Phân chia dữ liệu
    X, y,
    random_state=k,
    test_size=1/3.0
  )

  Z = DecisionTreeClassifier(
    criterion='gini', #Dựa trên chỉ số gini
    random_state=42, #Với random state = 42, chúng ta nhận được cùng một tập dữ liệu huấn luyện và thử nghiệm trên các lần thực thi 
    #khác nhau, nhưng lần này tập dữ liệu huấn luyện và thử nghiệm khác với trường hợp trước với random state = 0
    max_depth=10,
    min_samples_leaf=5 #nút nhánh có ít nhất là 5 con (không phải nút lá)
  )

  ZZ = BaggingClassifier( #Tập hợp mô hình
    base_estimator=Z, #Dùng giải thuật DT
    n_estimators=10, #Số lượng ước tính của base_estimator trong tập hợp
    random_state=42
  )

  ZZ.fit(X_train, y_train) #Xây dựng mô hình theo bagging 
  yy = ZZ.predict(X_test) #Dự đoán nhãn của tập kiểm tra
  YY.append(accuracy_score(y_test, yy)) #Thêm vào mảng #Tính độ chính xác cho ptu trong tap kiem tra
  i = i+1
  acc = accuracy_score(y_test, yy)*100
  print("Do chinh xac tong the lap " , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100))#round: làm tròn số gần nhất VD: 5.4->5 5.5->6
#round(sum(YY) / len(YY) * 100): tổng độ 9xac qua 10 lần lập
#ko để chữ string là in ra bị lỗi
print("Gini, Holdout, dùng baging: " + k)



# ================================================

# ====================KNN=========================
print("====================KNN=========================")

# hold out, dùng baging
i=0
YY = []
for k in range(N):#khoảng
  X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    random_state=k,
    test_size=1/3.0
  )

  Z = KNeighborsClassifier() #giá trị mặc định n_neighbors int, default=5 (5 láng giềng)
  
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
  Z = GaussianNB() #Xây dựng mô hình theo phân phối Gauss

  ZZ = BaggingClassifier(
    base_estimator=Z,
    n_estimators=10,
    random_state=42)

  ZZ.fit(X_train, y_train)
  yy = ZZ.predict(X_test)
  YY.append(accuracy_score(y_test, yy))
  i = i+1
  acc = accuracy_score(y_test, yy)*100
  print("Do chinh xac tong the lap " , i , ":", acc, "%")

k = str(round(sum(YY) / len(YY) * 100, 2))
print("Hold out, baging: " + k)