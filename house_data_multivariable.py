import pandas as pd
import numpy as np
import seaborn as sns
import statsmodels.api as sn
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


df = pd.read_csv("House_Price.csv")

#df_describe = df.describe().T #shows various info about dataframe you imported, useful for identifying outliers

#sns.jointplot(x="n_hot_rooms", y = "price", data = df) #scatterplot
#sns.jointplot(x="rainfall", y = "price", data = df) #scatterplot
#sns.countplot(x="airport",data=df) #bar graph
#sns.countplot(x="waterbody",data=df) #bar graph
#sns.countplot(x="bus_ter",data=df) #bar graph
#print(df.info())
'HANLING OUTLIERS'
upper_limit = (np.percentile(df.n_hot_rooms,[99])[0]) #view 99th percentile value, NOT the highest value, 0 indexes the first element of array, which will be the number

#upper = df[(df.n_hot_rooms > upper_limit)]
df.loc[(df.n_hot_rooms > 3*upper_limit, "n_hot_rooms")] = 3*upper_limit
        #limits which nums to show             # sets those nums equal to 3*99th percentile value
lower_limit = (np.percentile(df.rainfall,[1])[0])
#lower = df[(df.rainfall < lower_limit)]
df.loc[(df.rainfall < 0.3*lower_limit, "rainfall")] = 0.3*lower_limit
        #selects all nums below 1st percentile   #multiplies 1st percentile value by 0.3 and sets that as the new value for the lowest outlier
'-------------------------------------------------------------------'
###sns.displot(x=df['crime_rate'])
###sns.displot(x=df['price'])

df.crime_rate = np.log(1+df.crime_rate)
#sns.jointplot(y="crime_rate",x="price", data=df)

df["avg_dist"] = (df.dist1+df.dist2+df.dist3+df.dist4)/4
#combining variables that are redundant into one variable that is easier to use

del df["dist1"],df["dist2"],df["dist3"],df["dist4"],df["bus_ter"]
#deleting redundant variables that can be reduced into one

df.n_hos_beds = df.n_hos_beds.fillna(df.n_hos_beds.mean())
#fills in the missing data in hospital beds with the mean of the data


df = pd.get_dummies(df)
#function that creates dummy variables so that non-numerical data can also be analyzed
del df["airport_NO"]
#deletes redundant dummy variables because the same info is conveyed with one less dummy var


del df["parks"]
#deleting park variable due to its extremely high collinearity with air_quality which could cause multicollinearity issues later
#in general, delete variables that either have multicollinearity issues or variables that have very low correlation with other variables, 
#however, because there are already barely any variables in the dataset, we'll keep all the vars


df_head = df.head()
df_describe = df.describe().T
df_correlation = df.corr()
#allows you to view how the data affects each other in a grid format, negative means as one goes up, the other goes down, positive means they both show same trends

#df['flag'] = df['crime_rate'].apply(lambda x: round(x))
#print(df['flag'].value_counts())
#creates test-train datasets
#x_train, x_test, y_train, y_test = train_test_split(df['crime_rate'], df['price'],test_size=0.2,random_state=100,stratify=df['flag'])
'--------------------------------------------------'
x_train, x_test, y_train, y_test = train_test_split(df[['crime_rate','room_num']], df['price'],test_size=0.2,random_state=100)

df_train = pd.DataFrame(columns=['crime_rate','room_num','price'])
df_train[['crime_rate','room_num']] = x_train
df_train['price'] = y_train 


df_test = pd.DataFrame(columns=['crime_rate','room_num','price'])
df_test[['crime_rate','room_num']] = x_test
df_test['price'] = y_test

df_train = df_train.dropna()
#df_train.dropna(inplace=True)
df_test = df_test.dropna()

x1 = df_train[['crime_rate','room_num']].values
y1 = df_train['price'].values
#print(x1)
lm = LinearRegression().fit(x1,y1)
#a = lm.predict(x1)
#sns.scatterplot(x=x1,y=y1,data=df)
#sns.lineplot(x=x1,y=a,color="red")

x2 = df_test[['crime_rate','room_num']].values
y2 = df_test['price'].values
b = lm.predict(x2)
#sns.scatterplot(x=x2[:,0],y=y2)
#sns.lineplot(x=x2[:,0],y=b,color='red')
plt.xlabel('Num of Rooms and Crime Rate')
plt.ylabel('Price')
plt.title('KNN: Room Num and Crime Rate vs Price')
#print(lm.coef_)
#print(lm.intercept_)
#sns.displot(x=df['crime_rate'])
#sns.displot(x=df['price'])

result = pd.DataFrame(index=['Linear Regression','KNN'])
result['Mean Absolute Error'] = None
result['Mean Squared Error'] = None
result.loc['Linear Regression','Mean Absolute Error'] = mean_absolute_error(y2,b)
result.loc['Linear Regression','Mean Squared Error'] = mean_squared_error(y2,b)

#creates data using the OLS method with y-variable as price and x variable as room_num as defined in line above
#print(lm.summary())
#prints the data
'HYPERTUNING K VALUE'
num = float('inf')
final_k = 0
for i in range(1,11):
    knn_model = KNeighborsRegressor(n_neighbors=i)
    knn_model.fit(x1,y1)
    z = knn_model.predict(x2)
    if mean_squared_error(y2,z) <= num:
        num = mean_squared_error(y2,z)
        final_k = i
print(final_k)

knn_model = KNeighborsRegressor(n_neighbors=final_k)
knn_model.fit(x1,y1)
#z = knn_model.predict(c).ravel()

z = knn_model.predict(x2)
sns.scatterplot(x=x2[:,0],y=y2)
sns.lineplot(x=x2[:,0],y=z,color='green')

result.loc['KNN','Mean Absolute Error'] = mean_absolute_error(y2,z)
result.loc['KNN','Mean Squared Error'] = mean_squared_error(y2,z)
print(result)



'''
x = df[["room_num"]]
y = df["price"]
lm2 = LinearRegression()
lm2.fit(x,y)
#print(lm2.intercept_,lm2.coef_)
#print(lm2.predict(x))
z = lm2.predict(x) #predicts y-values based on the given x-values
#help(jointplot)
sns.scatterplot(x=df["room_num"],y=df["price"],data=df) #plots data as dots
sns.lineplot(x=df["room_num"],y=z,color="red")  #creates line of prediction based on predicted y-values
'''


'''
FIXES:
- missing values in n_hos_rooms
- outliers in crime rate
- outliers in n_hot_rooms and rainfall
- bus_ter has only one value in it (useless)

- dist1, dist2, dist3, dist4 variables (shows the distance from a job) 
are redundant, replace with a dist variable that takes the mean 
of all of them

note: for outliers, determine the 1 and 99 percentile of data using
np.percentile 
'''