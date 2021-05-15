import pandas as pd
import numpy as np
import re
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics
from sklearn import tree
from sklearn.datasets import load_iris
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from six import StringIO
from IPython.display import Image
import pydotplus


R = 496

df = pd.read_csv('Data.csv')
nan_value = float("NaN")
df.replace('',nan_value,inplace=True)
df.dropna(inplace=True)

symbol = (df.iloc[0, 3])[0]
df['Value'] = df['Value'].str[1:]

df.replace(' ', '', regex=True, inplace=True)
df.replace('Lac','|',regex=True,inplace=True)
df.replace('Cr','00|',regex=True,inplace=True)
df.replace(symbol,'', regex=True, inplace=True)
for i in range(R):
    value = df.iloc[i,1]
    res = re.sub("\D","",value)
    df.iloc[i,1] = res
for i in range(R):
    C = 0
    value = df.iloc[i,2]
    for j in range(len(value)):
        if value[j]=='(':
            C = j
    value = value[:C]
    df.iloc[i,2] = value
for i in range(R):
    C = 0
    value = df.iloc[i, 2]
    for j in range(len(value)):
        if value[j] == '-':
            C = j
    value = value[C:]
    df.iloc[i, 2] = value
for i in range(R):
    C = 0
    value = df.iloc[i, 3]
    for j in range(len(value)):
        if value[j]=='|':
            C = j
    value = value[:C]
    df.iloc[i, 3] = value
for i in range(R):
    C = 0
    value = df.iloc[i, 3]
    for j in range(len(value)):
        if value[j] == '-':
            C = j
    value = value[C:]
    df.iloc[i, 3] = value
df.replace('sq.ft.','',regex=True,inplace=True)
df.replace('-', '', regex=True, inplace=True)
df.replace(',', '', regex=True, inplace=True)
df['Value'] = pd.to_numeric(df['Value'])
for i in range(R):
    if df.iloc[i,3]<=10:
        df.iloc[i,3]*=100
df['Specification'] = pd.to_numeric(df['Specification'])
df['Area'] = pd.to_numeric(df['Area'])

df.replace('', nan_value, inplace=True)
df.dropna(inplace=True)

X = df[['Specification','Area']]
Y = df['Value']

#Regression fit
regr = linear_model.LinearRegression()
regr.fit(X,Y)
#Regression coefficient
R_cf = regr.coef_
print('Linear Regression Coefficient: ',R_cf)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
regressor = RandomForestRegressor(n_estimators=20, random_state=0)
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


df['Value'] = pd.to_numeric(df['Value'],downcast='integer')
Y = df['Value'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

spec = input('Specification: ')
area = input('Area: ')

inp = pd.DataFrame({'Specification':[spec],'Area':[area]})
preds = inp[['Specification','Area']]
print("Linear regression prediction: "+str(regr.predict(preds))+"lac inr")
print("Random forest classifier prediction: "+str(regressor.predict(preds))+"lac inr")
print("Decision tree prediction: " + str(clf.predict(preds))+"lac inr")
