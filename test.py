import numpy as np
import pandas as pd

dataset = pd.read_csv('./Wine_Quality_Data.csv') 
X = dataset.iloc[:,:-1].values 
Y  = dataset.color.map({'white':0, 'red':1})

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=4) 

forest_reg = DecisionTreeClassifier(random_state=42)
forest_reg.fit(x_train, y_train)

y_pred = forest_reg.predict(x_test) 

accuracy = confusion_matrix(y_test,y_pred) 
print(accuracy)
print(accuracy_score(y_test,y_pred))