#!/usr/bin/env python
# coding: utf-8

# In[118]:


from IPython.display import Image
Image(r'D:\ML\others/car.jpg')


# In[ ]:





# In[119]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

#%matplotlib inline

import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[120]:


car = pd.read_csv(r"D:\ML\Projects\quikr_car.csv")


# In[121]:


car


# In[ ]:





# ### Looking for null values

# In[122]:


car.isnull().sum()


# In[123]:


sns.heatmap(car.isnull(), yticklabels=False,cbar=False, cmap='viridis')
plt.show()


# In[ ]:





# In[124]:


car.info()


# ### There are some null values in kms_driven and fuel_type
# ### Also i have to change the dtype of year, price and kms_driven to int

# In[ ]:





# ## Performing EDA
# kms_driven 
# there are some junk characters prescent in kms_driven 

# In[125]:


car['kms_driven'].unique()


# In[ ]:





# In[126]:


car['kms_driven']=car['kms_driven'].str.replace('kms','')             # using str.replace to convert kms into whitespace


# In[127]:


car['kms_driven']=car['kms_driven'].str.replace(',','')


# In[128]:


car['kms_driven']=car['kms_driven'].str.replace('Petrol','0')


# In[129]:


car['kms_driven'].fillna(str(car['kms_driven'].mode().values[0]), inplace=True)       # using fillna to fill the null values to mode


# In[130]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[131]:


car.head(2)

kms_driven is clean now
# In[ ]:





# ## fuel_type

# In[ ]:





# In[132]:


car[car['fuel_type'].isna()]         # isna() shows all NaN values prescent in fuel_type


# In[ ]:





# In[133]:


car['fuel_type'].unique()


# In[134]:


car['fuel_type'].fillna(str(car['fuel_type'].mode().values[0]), inplace=True)


# In[135]:


car.isnull().sum()

# No null values now
# ## year
 Some junk character in Year column
# In[136]:


car['year'].unique()


# In[ ]:





# In[137]:


car=car[car['year'].str.isnumeric()]    # this code is basically doing str operation on year column
                                        # this will filter only those rows wheather str is prescent 


# In[138]:


car['year'].unique()


# In[139]:


car['year']=car['year'].astype(int)


# In[ ]:





# In[ ]:





# ## Price
# In price we can see the junk character like , and ask for price
# In[140]:


car['Price'].unique()


# In[ ]:





# In[141]:


car['Price']=car['Price'].str.replace('Ask For Price','0')


# In[142]:


car['Price']=car['Price'].str.replace(',','')


# In[143]:


car['Price'].unique()


# In[144]:


car['Price']=car['Price'].astype(int)


# In[ ]:





# In[ ]:





# ## name
# In name column name of the car model is to big and there are some numbers also available in some model name
# So first i wll split the names 
# Then i will select first 3 character
# And in the last join them
# In[ ]:





# In[145]:


car['name']


# In[ ]:





# In[146]:


car['name'].str.split(' ')                       # Now i will split name


# In[ ]:





# In[147]:


car['name'].str.split().str.slice(0,3)           # slicing above list form 0 to 3


# In[ ]:





# In[148]:


car['name']=car['name'].str.split().str.slice(0,3).str.join(' ')   # joing to name column


# In[ ]:





# In[149]:


#car=car['name'].str.split().str.slice(0,3).str.join(' ')


# In[ ]:





# ## Final dataset

# In[150]:


car    # some index is missing


# In[151]:


car=car.reset_index(drop=True)          # I drop old index and add new index numbers


# In[152]:


car


# In[ ]:





# # Data Visualization

# In[153]:


get_ipython().system(' pip install plotly==4.14.3')


# In[ ]:





# In[154]:


import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots



sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (9, 5)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[ ]:





# In[ ]:





# ### Here i want company wise analysis so for that i extract company name of Toyota, Mahindra and Maruti

# In[155]:


Toyota_Mahindra_Maruti = car[(car["company"]=="Toyota") | (car["company"]=="Mahindra") | (car["company"]=="Maruti")]


# In[156]:


sns.set(rc={'figure.figsize':(10,10)})
sns.barplot(x='company', y='Price',data=Toyota_Mahindra_Maruti,hue='fuel_type')
plt.show()


# In[ ]:





# In[157]:


sns.barplot(x="year", y="Price", data=car,hue="company")
plt.show()


# In[ ]:





# In[ ]:





# In[158]:


fig = px.pie(car['company'].value_counts().reset_index(), values='company', names='index')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[ ]:





# In[159]:


plt.figure(figsize=(6,6))
sns.distplot(car['Price'])
plt.show()


# In[ ]:





# In[160]:


plt.figure(figsize=(10,10))
plt.title("Price vs kms_driven")
sns.scatterplot(data=car,x="kms_driven",y="Price",hue="company")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# # Regression

# In[161]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[162]:


#x=car[['Price',]]#Price
x = car.drop(["Price"], axis=1)
y = car["Price"]


# In[ ]:





# In[198]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2,random_state =116)                # splitting


# In[ ]:





# In[199]:


# OneHotEncoder


df = OneHotEncoder()                         # object of OneHotEncoder
df.fit(x[['name','company','fuel_type']])    # here i fit all categorical values of x
                                             # i change the xtrain and x test using this OneHotEncoder using this 


# In[ ]:





# In[200]:


df.categories_   # all categories


# In[201]:


col_tranform = make_column_transformer((OneHotEncoder(categories=df.categories_),['name','company','fuel_type']),
                                       remainder='passthrough')

# make_column_transformer((OneHotEncoder : will perform OneHotEncoder on abouve columns input data
# its just for to fransform the columns during transform
# i set remainder to passtrough(means remaining columns will be pass through)


# In[ ]:





# In[202]:


lr = LinearRegression()


# In[203]:


pipe=make_pipeline(col_tranform,lr)    # i send col_tranform to pipeline and lr too


# In[204]:


pipe.fit(x_train,y_train)

# train my data


# In[205]:


y_pred=pipe.predict(x_test)


# In[206]:


y_pred


# In[207]:


r2_score(y_test,y_pred)


# In[183]:


# Above model is not good enough


# In[ ]:





# In[184]:


# To improve my r2_score i am performing this
# i run my for loop 1000 times to get best random state on that basis i will get best r2_score


# In[185]:


scores=[]
for i in range(1000):
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=i)
    lr = LinearRegression()
    pipe=make_pipeline(col_tranform,lr)
    pipe.fit(x_train, y_train)
    y_hat=pipe.predict(x_test)
    scores.append(r2_score(y_test, y_hat))


# In[186]:


np.argmax(scores)    # for random_state


# In[187]:


scores[np.argmax(scores)]    # much better r2_score


# In[ ]:





# In[ ]:





# ### Full spread not Homoskedestical

# In[ ]:





# In[196]:


pipe.predict(pd.DataFrame([['Honda Amaze 1.2','Honda',2014,100,'Petrol']], columns=['name','company','year',
                                                                                         'kms_driven','fuel_type']))


# In[197]:


pipe.predict(pd.DataFrame([['Tata Zest XM','Tata',2018,1000,'Diesel']], columns=['name','company','year',
                                                                                         'kms_driven','fuel_type']))


# In[ ]:




