#!/usr/bin/env python
# coding: utf-8

# #  Real Estate price predicter

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing['CHAS'].value_counts()


# In[6]:


housing['ZN'].value_counts()


# In[7]:


housing.describe()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


housing.hist(bins=50, figsize=(20,15))


# # #  Train test spliting

# In[11]:


#learning purpose
import numpy as np
def split_train_test(data ,test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data)* test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size :]
    return data.iloc[train_indices], data.iloc[test_indices]


# In[12]:


#train_set , test_set = split_train_test(housing,0.2)


# In[13]:


#print(f"Rows in train set : {len(train_set)}\nRows in test set :{len(test_set)}\n")


# In[14]:


from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set : {len(train_set)}\nRows in test set :{len(test_set)}\n")


# In[15]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, test_index in split.split(housing,housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[16]:


strat_train_set['CHAS'].value_counts()


# In[17]:


strat_test_set['CHAS'].value_counts()


# In[18]:


housing = strat_train_set.copy()


# In[19]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM","ZN","LSTAT"]
scatter_matrix(housing[attributes], figsize = (12,8)) 


# ## Trying Attribute Combination

# In[20]:


housing['TAXRM'] = housing['TAX']/housing["RM"]


# In[21]:


housing.head()


# In[22]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[23]:


housing.plot(kind="scatter",x = "TAXRM",y = "MEDV", alpha=0.8)


# In[24]:


housing = strat_train_set.drop("MEDV",axis=1)
housing_labels = strat_train_set["MEDV"].copy()


# ## In Case of Missing Attributes

# In[25]:


# in case of missing attributes there is 3 ways to deal with it
# 1.  delet missing data points
# 2.  delet whole column
# 3.  set the value (mean , 0, median)


# In[26]:


# for case 1
# a=dropna(subset=['RM']).shape
# a.shape


# In[27]:


#for case 2
# housing.drop("RM",axis=1).shape


# In[28]:


# for case 3
# median=housing["RM"].median()
# housing["RM"].fillna(median)


# In[29]:


housing.describe() #before we start imputer


# In[30]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[31]:


imputer.statistics_


# In[32]:


X = imputer.transform(housing)


# In[33]:


housing_tr = pd.DataFrame(X, columns=housing.columns)


# In[34]:


housing_tr.describe()


# ## Scikit-learn design

# In[35]:


# 1. Estimator
# 2. transformer
# 3. predictor


# ## Creating a pipeline

# In[36]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scalar',StandardScaler()),
])


# In[37]:


housing_num_tr = my_pipeline.fit_transform(housing)


# In[38]:


housing_num_tr.shape


# ## Selecting a desired model for Real Estate company

# In[39]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr,housing_labels)


# In[40]:


some_data = housing.iloc[:5]


# In[41]:


some_labels = housing_labels.iloc[:5]


# In[42]:


prepared_data = my_pipeline.transform(some_data)


# In[43]:


model.predict(prepared_data)


# In[44]:


list(some_labels)


# ## Evaluating the model

# In[45]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)


# In[46]:


rmse


# ## Using better Evaluation with cross Validation

# In[47]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr , housing_labels, scoring="neg_mean_squared_error", cv = 10)
rmse_scores = np.sqrt(-scores)


# In[48]:


rmse_scores


# In[49]:


def print_scores(scores):
    print("scores :",scores)
    print("mean :",scores.mean())
    print("standard daviation :",scores.std())


# In[50]:


print_scores(rmse_scores)


# ## Saving the Model

# In[51]:


from joblib import dump, load
dump(model, 'Realstate.joblib')


# ## Test model on test data

# In[52]:


X_test = strat_test_set.drop("MEDV", axis=1)
Y_test = strat_test_set["MEDV"].copy()
X_test_preparerd = my_pipeline.transform(X_test)
final_predictions = model.predict(X_test_preparerd)
final_mse = mean_squared_error(Y_test , final_predictions)
final_rmse = np.sqrt(final_mse)


# In[53]:


final_rmse


# In[56]:


prepared_data


# In[ ]:




