from matplotlib import colors
from numpy import nan
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

Movies =  pd.read_csv("C:/Users/ahmed/source/repos/PythonApplication1/tmdb-movies.csv")
drop = ['id','imdb_id','original_title','homepage','tagline','overview','release_date','keywords', 'director']
Movies.drop(drop, axis = 1 , inplace = True)
Movies.drop_duplicates()
nan_col = ['budget','revenue','revenue_adj','budget_adj']
Movies[nan_col] = Movies[nan_col].replace({'0':np.nan, 0:np.nan})
missing_col = ['budget']
for i in missing_col:
 Movies.loc[Movies.loc[:,i].isnull(),i]=Movies.loc[:,i].mean()
missing_col1 = ['revenue']
for i in missing_col1:
 Movies.loc[Movies.loc[:,i].isnull(),i]=Movies.loc[:,i].mean()
missing_col2 = ['revenue_adj']
for i in missing_col2:
 Movies.loc[Movies.loc[:,i].isnull(),i]=Movies.loc[:,i].mean()
missing_col3 = ['budget_adj']
for i in missing_col3:
 Movies.loc[Movies.loc[:,i].isnull(),i]=Movies.loc[:,i].mean()
Movies.dropna(how ='any',inplace=  True)
changer = LabelEncoder()
Movies["genres"] = changer.fit_transform(Movies["genres"])
Movies["production_companies"] = changer.fit_transform(Movies["production_companies"])
Movies["cast"] = changer.fit_transform(Movies['cast'])
scaler = MinMaxScaler()
Movies_scale = pd.DataFrame(scaler.fit_transform(Movies), columns= Movies.columns)      
x_data =  Movies_scale[['popularity','budget','revenue','runtime','vote_count','vote_average','production_companies','cast','genres']]
y_data = Movies_scale[['budget_adj', 'revenue_adj']]
Y_data = y_data['revenue_adj'] - y_data['budget_adj']
x_train,x_test,y_train,y_test = train_test_split(x_data,Y_data,test_size= 0.20,random_state= 20)
reg = LinearRegression()
reg.fit(x_train,y_train)
y_predict = reg.predict(x_test)
print(reg.score(x_test,y_test))
plt.scatter(y_test,y_predict)
plt.plot(y_test,y_predict)
plt.show()
