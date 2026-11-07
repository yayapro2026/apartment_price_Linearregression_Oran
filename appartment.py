import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
df=pd.read_csv('immobilier_oran_donnees.csv')
k=1
j=1
for i in list(set(df["Localisation"])):
    df.replace(to_replace=i,value=j,inplace=True)
    j+=1
for i in list(set(df["Type_Bien"])):
    df.replace(to_replace=i,value=k,inplace=True)
    k+=1
X=df[["Type_Bien","Localisation","Superficie_m2"]]
Y=df["Prix_Target_DA"]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,train_size=0.8,random_state=42)
model=LinearRegression()
model.fit(X_train,Y_train)
plt.figure(figsize=(7,5))
plt.scatter(Y,model.predict(X),colorizer="purple")
plt.plot(Y,Y)
print(np.sqrt(mean_squared_error(Y_test,model.predict(X_test))))
print(np.mean(df["Prix_Target_DA"]))
plt.show()