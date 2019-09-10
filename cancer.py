import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
s=pd.read_csv('breast-cancer-data.csv')
df=pd.DataFrame(s)
# print(df.head())
# sns.heatmap(s.corr())
# plt.show()
print(df.drop('id',axis=1,inplace=True))
# print(df.head())
# print(df.columns)
df['diagnosis']=df.diagnosis.map({'M':0,'B':1})#to change the values of diagnosis table
# print(df.head())
x=df[[  'radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']]
y=df['diagnosis']

x_trained, x_test , y_trained, y_test = train_test_split(x,y,test_size=.4,random_state=101)
# print(x_test)
# print(y_test)
lm = LinearRegression()#object created in linear regression
lm.fit(x_trained,y_trained)#method calling
pp=(lm.coef_)#it will give the corerelation of yearly amount spent with every other collumn value
print(pp)
l=pd.DataFrame(pp,index=['radius_mean', 'texture_mean', 'perimeter_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst'],columns=["coef"])
print(l)
#
#
sns.countplot(x="diagnosis",data=df)
plt.show()
#
#
print(df.corr())#to create the corelation
sns.heatmap(data=df,annot=True)
plt.show()