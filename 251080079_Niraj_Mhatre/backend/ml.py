import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df=pd.read_hdf("metr-la.h5")
df=df.ffill()

X=pd.DataFrame({
    "f1":df.mean(axis=1),
    "f2":df.std(axis=1),
    "f3":df.max(axis=1)
})

y=df.mean(axis=1)

Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,test_size=0.2,random_state=42)

model=LinearRegression()
model.fit(Xtrain,ytrain)

with open("model.pkl","wb") as f:
    pickle.dump(model,f)
