# Decision-Tree

import pandas as pd
import matplotlib.pyplot as plt

wine = pd.read_csv('wined.csv')
wine

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

x = wine[['Acl','Alcohol','Malic.acid','Ash']]
y = wine[['Wine']]


wine['Acl']=le.fit_transform(wine['Acl'])
wine['Alcohol']=le.fit_transform(wine['Alcohol'])
wine['Malic.acid']=le.fit_transform(wine['Malic.acid'])
wine['Ash']=le.fit_transform(wine['Ash'])
wine.head()

from sklearn import tree
model = tree.DecisionTreeClassifier()
model.fit(x,y)
model.predict([[1,65,49,18]])

plt.figure(figsize=(15,10))
tree.plot_tree(model,filled=True)
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)
model.fit(x_train,y_train)
y_predicted = model.predict(x_test)
y_predicted

from sklearn.metrics import accuracy_score
sc = accuracy_score(y_predicted,y_test)*100
sc

from sklearn.metrics import mean_squared_error
mean_squared_error(y_predicted,y_test)*100
