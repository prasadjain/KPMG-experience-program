#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


transactions = pd.read_excel("KPMG.xlsx",sheet_name="Transactions")


# In[3]:


transactions.head()


# In[4]:


transactions.isna().sum()


# In[5]:


# printing the range of dates of the dataset
print('Min : {}, Max : {}'.format(min(transactions.transaction_date), max(transactions.transaction_date)))


# In[6]:


from datetime import datetime  
from datetime import timedelta 


# In[7]:


# adding a day to the most recent date
last_date = max(transactions.transaction_date) + timedelta(1)


# In[8]:


# calculating the recency, Frequency and Monetary value columns from the transaction dataset.
rfm = transactions.groupby('customer_id').agg({
    'transaction_date' : lambda x: (last_date - x.max()).days,
    'transaction_id' : 'count', 
    'list_price' : 'sum'})


# In[9]:


rfm.rename(columns = {'transaction_date' : 'Recency', 
                      'transaction_id' : 'Frequency', 
                      'list_price' : 'Monetary'}, inplace = True)


# In[10]:


rfm.head()


# In[11]:


rfm.describe()


# In[12]:


import seaborn as sns
from matplotlib import pyplot as plt


# In[13]:


# visualising each column to see whether it is normally distributed. 
sns.distplot(rfm['Recency'])


# In[14]:


# log  transformation of the recency column
recency_log=np.log(rfm['Recency'])
sns.distplot(recency_log)


# In[15]:


rfm.head()


# In[16]:


sns.distplot(rfm['Frequency'])


# In[17]:





# In[18]:


sns.distplot(rfm['Monetary'])


# In[19]:





# In[20]:


rfm.head()


# In[21]:





# In[22]:





# In[23]:





# In[24]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[25]:


#standard scaling all the columns before applying K-means clustering
scaler.fit(rfm)
norm_rfm = pd.DataFrame(scaler.transform(rfm))


# In[26]:


norm_rfm.head()


# In[27]:


norm_rfm.describe()


# In[28]:


from sklearn.cluster import KMeans


# In[29]:


sse={}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(norm_rfm)
    sse[k] = kmeans.inertia_


# In[30]:


plt.title('The Elbow Method')
plt.xlabel('k'); plt.ylabel('SSE')
sns.pointplot(x=list(sse.keys()), y=list(sse.values()))
plt.show()


# In[31]:


#selecting n=2 as only 2 segments of customers were required
Kmeans = KMeans(n_clusters=2)


# In[32]:


Kmeans.fit(norm_rfm)


# In[33]:


cluster_labels = Kmeans.labels_


# In[34]:


rfm_k2 = rfm.assign(Cluster = cluster_labels)


# In[35]:


rfm_k2.groupby(['Cluster']).agg({
'Recency': 'mean',
'Frequency': 'mean',
'Monetary': ['mean', 'count'],
}).round(0)


# In[36]:


cust_demo = pd.read_excel("KPMG.xlsx",sheet_name="CustomerDemographic")
cust_address = pd.read_excel("KPMG.xlsx",sheet_name="CustomerAddress")


# In[37]:


cust_demo.head()


# In[38]:


cust_demo.isna().sum()


# In[39]:


cust_demo.drop(['first_name','last_name','default','job_title'],axis=1,inplace=True)


# In[40]:


cust_demo.isna().sum()


# In[41]:


customer = pd.merge(cust_address,cust_demo,how='left',left_on='customer_id',right_on='customer_id')


# In[42]:


customer.head(15)


# In[43]:


customer.drop(['address','postcode','country','DOB'],axis=1,inplace=True)


# In[ ]:





# In[44]:


customer = pd.merge(customer,rfm_k2,how='left',left_on='customer_id',right_on='customer_id')


# In[45]:


customer.isna().sum()


# In[46]:


customer.drop(['Recency','Frequency','Monetary'],axis=1,inplace=True)


# In[47]:


customer.head()


# In[48]:


customer.drop('customer_id',axis=1,inplace=True)


# In[49]:


customer.head()


# In[50]:


from sklearn.impute import SimpleImputer


# In[51]:


customer.isna().sum()


# In[52]:


# imputing the misssing values
si = SimpleImputer(missing_values=np.nan,strategy='constant',fill_value='No Information') 
customer['imp_job_industry'] = pd.DataFrame(si.fit_transform(customer[['job_industry_category']]),index=customer.index)


# In[53]:


siten = SimpleImputer(missing_values=np.nan,strategy='median')
customer['imp_tenure'] = pd.DataFrame(siten.fit_transform(customer[['tenure']]),index=customer.index)


# In[54]:


customer.isna().sum()


# In[55]:


customer.drop(['job_industry_category','tenure'],axis=1,inplace=True)


# In[56]:


customer.isna().sum()


# In[57]:


customer = customer.dropna()


# In[58]:


customer.isna().sum()


# In[59]:


customer.info()


# In[60]:


customer.shape


# In[61]:


X = customer.drop(['Cluster'],axis=1)
y = customer['Cluster']


# In[62]:


from sklearn.model_selection import train_test_split
Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)


# In[63]:


Xtrain = Xtrain.copy()
Xtest = Xtest.copy()
ytrain = ytrain.copy()
ytest = ytest.copy()


# In[64]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import MinMaxScaler


# In[65]:


Xtrain.head()


# In[ ]:





# In[66]:


# binning age 
kbdage = KBinsDiscretizer(n_bins=5,encode='ordinal',strategy='quantile')
Xtrain['kbinAge']=pd.DataFrame(kbdage.fit_transform(Xtrain[['Age']]),index=Xtrain.index)


# In[67]:


Xtest['kbinAge']=pd.DataFrame(kbdage.transform(Xtest[['Age']]),index=Xtest.index)


# In[68]:


Xtrain.drop('Age',axis=1,inplace=True)


# In[69]:


Xtest.drop('Age',axis=1,inplace=True)


# In[70]:


categorical_features = ['state','gender','wealth_segment','deceased_indicator','owns_car','imp_job_industry','kbinAge']


# In[71]:


ohe = OneHotEncoder(sparse=False,dtype=int,handle_unknown='ignore')


# In[72]:


Xcat=pd.DataFrame(ohe.fit_transform(Xtrain[categorical_features]),columns=ohe.get_feature_names(),index=Xtrain.index)


# In[73]:


Xtrain=pd.concat([Xtrain,Xcat],axis=1)


# In[74]:


Xtrain.columns


# In[75]:


Xcat = pd.DataFrame(ohe.transform(Xtest[categorical_features]),columns=ohe.get_feature_names(),index=Xtest.index)
Xtest = pd.concat([Xtest,Xcat],axis=1)


# In[76]:


Xtrain.drop(categorical_features,axis=1,inplace=True)


# In[77]:


Xtest.drop(categorical_features,axis=1,inplace=True)


# In[78]:


Xtrain.columns


# In[79]:


#setting up pipeline for preprocessing 
numeric_features = ['property_valuation','past_3_years_bike_related_purchases','imp_tenure']


# In[80]:


mms=MinMaxScaler()


# In[81]:


Xnum = pd.DataFrame(mms.fit_transform(Xtrain[numeric_features]),columns=['mms_'+x for x in numeric_features],index=Xtrain.index)
Xtrain=pd.concat([Xtrain,Xnum],axis=1)


# In[82]:


Xnum = pd.DataFrame(mms.transform(Xtest[numeric_features]),columns=['mms_'+x for x in numeric_features],index=Xtest.index)
Xtest=pd.concat([Xtest,Xnum],axis=1)


# In[83]:


Xtrain.drop(numeric_features,axis=1,inplace=True)
Xtest.drop(numeric_features,axis=1,inplace=True)


# In[84]:


Xtrain.head()


# In[85]:


from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()


# In[86]:


lr.fit(Xtrain,ytrain)


# In[87]:


ypred = lr.predict(Xtest)


# In[88]:


from sklearn import metrics
from sklearn.metrics import confusion_matrix


# In[89]:


print(metrics.accuracy_score(ytest,ypred))


# In[90]:


confusion_matrix(ytest,ypred)


# In[91]:


from sklearn.naive_bayes import GaussianNB


# In[92]:


gnb = GaussianNB()
ypred = gnb.fit(Xtrain,ytrain).predict(Xtest)


# In[93]:


print(metrics.accuracy_score(ytest,ypred))


# In[94]:


confusion_matrix(ytest,ypred)


# In[95]:


from sklearn import model_selection
from sklearn.ensemble import GradientBoostingClassifier


# In[96]:


clf=GradientBoostingClassifier()


# In[97]:


clf.fit(Xtrain,ytrain)
ypred=clf.predict(Xtest)


# In[98]:


print(metrics.accuracy_score(ytest,ypred))


# In[99]:


confusion_matrix(ytest,ypred)


# In[ ]:




