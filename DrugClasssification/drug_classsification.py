import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

label_encoding = LabelEncoder()

data = pd.read_csv("drug200.csv")
print(data)

data["Sex"] = label_encoding.fit_transform(data["Sex"])

bp_dict = {"HIGH": 2, "NORMAL": 1, "LOW": 0}
cholesterol_dict = {"HIGH": 1, "NORMAL": 0}
data = pd.get_dummies(data, columns=['Sex', 'BP', 'Cholesterol'], dtype=float)
object = StandardScaler()
# print(data[data["Age"].isna()])
data["Age"] = object.fit_transform(data[["Age"]])
data["Na_to_K"] = object.fit_transform(data[["Na_to_K"]])
X = data.drop(columns=["Drug"])
Y = data['Drug']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=21, test_size=0.2)
print(len(X_train))
print(len(Y_train))

print(X.columns)
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
y_pred = decision_tree.predict(X_test)
print(y_pred)
score = accuracy_score(y_pred, Y_test)
y_train_pred=decision_tree.predict(X_train)
train_score = accuracy_score(y_train_pred, Y_train)
print(train_score)

# print(data)
