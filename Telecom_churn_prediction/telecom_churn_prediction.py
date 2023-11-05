import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("tele_com.csv")
df['TotalCharges'] = df['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()

df.drop(columns=["customerID"], inplace=True)
le = LabelEncoder()
categorical_columns = ['TotalCharges', 'MonthlyCharges', 'SeniorCitizen', 'tenure', 'gender', 'Churn', 'Partner',
                       'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
for i in categorical_columns:
    df[i] = le.fit_transform(df[i])
standard_scaler = StandardScaler()
df['MonthlyCharges'] = standard_scaler.fit_transform(df[['MonthlyCharges']])
df['TotalCharges'] = standard_scaler.fit_transform(df[['TotalCharges']])

X = df.drop(columns=["Churn"])
Y = df["Churn"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=3, test_size=0.3)
knn_neighbor_graph = {}
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, Y_train)
    y_pred = knn.predict(X_test)
    score = accuracy_score(y_pred, Y_test)
    knn_neighbor_graph[str(i)] = score
import matplotlib.pyplot as plt

plt.plot([int(i) for i in list(knn_neighbor_graph.keys())], list(knn_neighbor_graph.values()))
plt.xlim(1, 40)
plt.title("KNN Neighbor Classifier")
plt.show()

# from analysis it is found that k=29 is the best value for prediction
knn = KNeighborsClassifier(n_neighbors=29)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
score = accuracy_score(y_pred, Y_test)
print(score)
