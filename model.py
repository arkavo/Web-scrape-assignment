import pandas as pd
import re

R = 525

df = pd.read_csv('Data.csv')
symbol = (df.iloc[0, 3])[0]
#print(symbol)
df['Value'] = df['Value'].str[1:]

df.replace(' ', '', regex=True, inplace=True)
df.replace('Lac','|',regex=True,inplace=True)
df.replace('Cr','00|',regex=True,inplace=True)
#df.replace('-', '\0xE2',regex=True,inplace=True)
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
print(df.head)
