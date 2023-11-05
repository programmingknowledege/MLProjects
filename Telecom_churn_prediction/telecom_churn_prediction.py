import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

df = pd.read_csv("tele_com.csv")
print(df)

df['TotalCharges'] = df['TotalCharges'].apply(lambda x: pd.to_numeric(x, errors='coerce')).dropna()
# print(df)
from sklearn import preprocessing

df.drop(columns=["customerID"], inplace=True)
le = preprocessing.LabelEncoder()
categorical_columns = ['TotalCharges', 'MonthlyCharges', 'SeniorCitizen', 'tenure', 'gender', 'Churn', 'Partner',
                       'Dependents', 'PhoneService', 'MultipleLines',
                       'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                       'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaperlessBilling', 'PaymentMethod']
for i in categorical_columns:
    df[i] = le.fit_transform(df[i])
print(df.head())

standard_scaler = StandardScaler()
df['MonthlyCharges'] = standard_scaler.fit_transform(df[['MonthlyCharges']])
df['TotalCharges'] = standard_scaler.fit_transform(df[['TotalCharges']])

X = df.drop(columns=["Churn"])
Y = df["Churn"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=3, test_size=0.3)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, Y_train)
y_pred = knn.predict(X_test)
from sklearn.metrics import accuracy_score

score = accuracy_score(y_pred, Y_test)
print(score)
