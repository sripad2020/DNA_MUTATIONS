from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn

data=pd.read_csv('MY_DATASET/Lable_Dataset.csv')
print(data.columns)
print(data.info())
print(data.isna().sum())
print(data.describe())

lab=LabelEncoder()
for i in data.select_dtypes(include='object').columns.values:
    data[i]=lab.fit_transform(data[i])


for i in data.select_dtypes(include='object').columns.values:
    print(data[i].value_counts().values)

for i in data.columns.values:
    sn.boxplot(data[i])
    plt.show()

for i in data.select_dtypes(include='number').columns.values:
    if len(data[i].value_counts()) <=7:
        val=data[i].value_counts().values
        index=data[i].value_counts().index
        plt.pie(val,labels=index,autopct='%1.1f%%')
        plt.title(f'The PIE Chart information of {i} column')
        plt.show()

'''for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.distplot(data[i], label=f"{i}", color='blue')
        sn.distplot(data[j], label=f"{j}", color="orange")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''

'''for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.histplot(data[i], label=f"{i}", color='blue')
        sn.histplot(data[j], label=f"{j}", color="orange")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()

sn.scatterplot(data)
plt.show()


sn.pairplot(data)
plt.show()

sn.pairplot(data,hue='Mutation_Type')
plt.show()

for i in data.columns.values:
    for j in data.columns.values:
        sn.scatterplot(data=data,x=i,y=j,hue='Mutation_Type')
        plt.show()

for i in data.columns.values:
    for j in data.columns.values:
        sn.jointplot(data=data,x=i,y=j,hue='Mutation_Type')
        plt.show()'''

'''for i in data.select_dtypes(include='number').columns.values:
    for j in data.select_dtypes(include='number').columns.values:
        sn.lineplot(data[i], label=f"{i}", color='blue')
        sn.lineplot(data[j], label=f"{j}", color="orange")
        plt.title(f"ITS {i} vs {j}")
        plt.legend()
        plt.show()'''

'''plt.figure(figsize=(17, 6))
corr = data.corr(method='spearman')
my_m = np.triu(corr)
sn.heatmap(corr, mask=my_m, annot=True, cmap="Set2")
plt.show()'''

x=data[['Reference_Codon','Position']]
y=data['Mutation_Type']

x_train,x_test,y_train,y_test=train_test_split(x,y)
lr=LogisticRegression()
lr.fit(x_train,y_train)
print('The logistic regression: ',lr.score(x_test,y_test))

lgb=LGBMClassifier()
lgb.fit(x_train,y_train)
print('The LGB',lgb.score(x_test,y_test))

tree=DecisionTreeClassifier()
tree.fit(x_train,y_train)
print('Dtree ',tree.score(x_test,y_test))

rforest=RandomForestClassifier()
rforest.fit(x_train,y_train)
print('The random forest: ',rforest.score(x_test,y_test))

adb=AdaBoostClassifier()
adb.fit(x_train,y_train)
print('the adb ',adb.score(x_test,y_test))

grb=GradientBoostingClassifier()
grb.fit(x_train,y_train)
print('Gradient boosting ',grb.score(x_test,y_test))

bag=BaggingClassifier()
bag.fit(x_train,y_train)
print('Bagging',bag.score(x_test,y_test))


X=data[['Reference_Codon', 'Query_Codon']]
Y=pd.get_dummies(data['Mutation_Type'])
x_tran,x_tst,y_tran,y_tst=train_test_split(X,Y)


from keras.models import  Sequential
from keras.layers import Dense
import keras.activations,keras.losses,keras.optimizers
models=Sequential()
models.add(Dense(units=x_tst.shape[1],input_dim=x_tst.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.relu))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.tanh))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.tanh))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.tanh))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=x_tst.shape[1],activation=keras.activations.softmax))
models.add(Dense(units=Y.shape[1],activation=keras.activations.softmax))
models.compile(optimizer='adam',metrics='accuracy',loss=keras.losses.categorical_crossentropy)
histo=models.fit(x_tran,y_tran,batch_size=45,epochs=450)
plt.plot(histo.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(histo.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam rmsprop')
plt.legend()
plt.show()

models1=Sequential()
models1.add(Dense(units=x_tst.shape[1],input_dim=x_tst.shape[1],activation=keras.activations.relu))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.relu))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.relu))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.relu))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.tanh))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.tanh))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.tanh))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.softmax))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.softmax))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.softmax))
models1.add(Dense(units=x_tst.shape[1],activation=keras.activations.softmax))
models1.add(Dense(units=Y.shape[1],activation=keras.activations.softmax))
models1.compile(optimizer='rmsprop',metrics='accuracy',loss=keras.losses.categorical_crossentropy)
histori=models1.fit(x_tran,y_tran,batch_size=45,epochs=450)
plt.plot(histori.history['accuracy'], label='training accuracy', marker='o', color='red')
plt.plot(histori.history['loss'], label='loss', marker='o', color='darkblue')
plt.title('Training Vs  Validation accuracy with adam rmsprop')
plt.legend()
plt.show()