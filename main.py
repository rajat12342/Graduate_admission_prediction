
#Methods are created using def but they need to be called using objects
#class Dog():
    #def __init__(self, name):    #creates constructor(main method) of class Dog
    #    self.name = name

    #def speak(self): #self needs to be passed to know what instance the method is called on
    #    print("Hello I am ",self.name)

    #def change_age(self, age):
    #    self.age = age

#a = Dog("bob")  #Creating instance of class automatically calls constructor
#a.speak()
#b= Dog("rob")
#b.speak()

#print("\n",a.name)
#print(b.name)
#a.change_age(75)
#print(a.age)


import pandas
import pandas as pd

import seaborn as sns
from tensorflow import keras
#from tensorflow.keras import layers
from tensorflow.python.keras.layers import Dense
import tensorflow
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from tensorflow.keras import callbacks

import pickle

data = pandas.read_csv("Admission_Predict.csv", index_col=0)


#print(sns.regplot(x=data['GRE Score'], y=data['Chance of Admit ']))
print(data['GRE Score'].describe())

#print(sns.boxplot(y = data['CGPA']))

x = data[['CGPA','GRE Score']]
y = data['Chance of Admit ']

x = np.array(x)
y= np.array(y)
print(y)

model = LinearRegression()

model.fit(x,y)
r_squared = model.score(x,y)

print("Prediction: ",model.predict([[8.5,330]]))

print("coefficient of determination: ",r_squared)
print("coefficient of correlation", model.coef_)
print("y-intercept: ", model.intercept_)


r = np.corrcoef(data['GRE Score'], data['Chance of Admit '])
print('Pearson\'s r: ', r)




#print(sns.barplot(x=data['LOR '], y=data['Chance of Admit ']))

#print(sns.lmplot(x="GRE Score", y='CGPA', data=data, hue='Research'))

corr = data.corr()
fig, ax = plt.subplots(figsize=(8, 8))

#makes fancy colours
#colormap = sns.diverging_palette(220, 10, as_cmap=True)

#make a zero matrix with the same dimensions as the one inputted
#dropSelf = np.zeros_like(corr)

#np.triu_indicies_from returns the indicies of the upper right triangle and the following code sets them to true
#dropSelf[np.triu_indices_from(dropSelf)] = True


#make heatmap - mask parameter hides the indices that are in dropself as true
#sns.heatmap(corr, cmap=colormap, linewidths=.5, annot=True, fmt=".2f", mask=dropSelf)


sns.countplot(x = data['SOP'])



X = data.drop(['Chance of Admit '],axis=1)

print(X)

Y = data['Chance of Admit ']

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X,Y, test_size= 0.3, shuffle=False)

LinearModel = LinearRegression()

LinearModel.fit(Xtrain.values,Ytrain)

predictions = LinearModel.predict(Xtest)
Rsquared = LinearModel.score(Xtrain, Ytrain)
Rsquared1 = LinearModel.score(Ytest, predictions)

print("Linear regression root mean squared error: ", np.sqrt(mean_squared_error(Ytest,predictions)))
print("Linear regression mean absolute error: ", mean_absolute_error(Ytest,predictions))
print("Linear regression R-Squared: ", Rsquared1)

print(LinearModel.predict([[309,100,2,3,3,8.1,0]]))

import joblib

data = {'model': LinearModel}
with open('linearModel.pkl', 'wb') as file:
    pickle.dump(data, file)


with open('linearModel.pkl', 'rb') as file:
    data = pickle.load(file)

regression_loaded = data['model']

predict = np.array([[309,100,2,3,3,8.1,0]])

y_pred = regression_loaded.predict(predict)
print(y_pred)


from sklearn.ensemble import RandomForestRegressor

Randomforest = RandomForestRegressor(random_state=0)

Randomforest.fit(Xtrain.values, Ytrain)

predictions1 = Randomforest.predict(Xtest)
Rsquared1 = Randomforest.score(Xtrain, Ytrain)

print("Root mean squared error: ", np.sqrt(mean_squared_error(Ytest,predictions1)))
print("Rsquared: ", Rsquared1)

print(Randomforest.predict([[309,100,2,3,3,8.1,0]]))
'''


ModelNeural = keras.Sequential([layers.Dense(units=100 ,input_shape=[7]),
                                layers.Dense(units=50),
                                layers.Dense(units=1)])

ModelNeural.compile(optimizer = 'adam', loss='mse')

early_stopping = callbacks.EarlyStopping(min_delta = 0.001, patience = 5, restore_best_weights = True)

history = ModelNeural.fit(Xtrain,Ytrain, validation_data = (Xtest,Ytest), batch_size=150, epochs = 35, callbacks = early_stopping)



#ModelNeural.


#w, b = ModelNeural.weights

#print("Weight: {},  bias: {}".format(w,b))

history_df = pd.DataFrame(history.history)
history_df.loc[:,['loss','val_loss']].plot()
print("Minimum validation loss: {}".format(history_df['val_loss'].min()))

print(ModelNeural.evaluate(Xtest,Ytest))



arr = np.array([[309,100,2,3,3,8.1,0]])


testdf = pd.DataFrame({'nigeria' : [0,1,2], 'canada': [4,5,6]}, index=['big','boy','man'])
print(testdf)
print(ModelNeural.predict(arr))

'''



#plt.title('Distribution of Gre scores')
#plt.ylim(0,10)
plt.show()








