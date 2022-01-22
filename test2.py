from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,accuracy_score,recall_score,precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from pytictoc import TicToc

t = TicToc()

t.tic() 
x_train,x_test,y_train,y_test = train_test_split(X,Y,random_state=4) #★★★★

#forest_reg = DecisionTreeClassifier(random_state=42)
forest_reg = RandomForestRegressor(n_estimators=1000, bootstrap=True, criterion='mse', max_depth=None, max_leaf_nodes=None, max_features='auto')

# 다섯 폴드에서 훈련하면 총 (12+6)*5=90번의 훈련이 일어납니다.
forest_reg.fit(x_train, y_train)


print(forest_reg.score(x_train, y_train)) 
print(forest_reg.score(x_test, y_test)) 
t.toc() 

#y_pred = forest_reg.predict(x_test) #★★★★
#accuracy = confusion_matrix(y_test,y_pred) #★★★★
#print(accuracy)
#accuracy_score(y_test,y_pred)