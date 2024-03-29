#!/usr/bin/env python
# coding: utf-8

#  # Importing Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# # Loding the dataset

# In[2]:


df = pd.read_csv('hotel_booking.csv')


# # Exploratory Data Analysis and Data Cleaning

# In[3]:


df.head()


# In[4]:


df.tail()


# In[5]:


df.shape


# In[6]:


df.columns


# In[7]:


df.info()


# In[8]:


df['reservation_status_date'] = pd.to_datetime(df['reservation_status_date'])


# In[9]:


df.info()


# In[10]:


df.describe(include = 'object')


# In[11]:


for col in df.describe(include = 'object').columns:
    print(col)
    print (df[col].unique())
    print('-'*50)


# In[12]:


df.isnull().sum()


# In[13]:


df.drop(['company', 'agent'], axis = 1, inplace = True)
df.dropna(inplace = True)


# In[14]:


df.isnull().sum()


# In[15]:


df.describe()


# In[16]:


df['adr'].plot(kind = 'box')


# In[17]:


df =df[df['adr']<5000]


# In[18]:


df.describe()


# # Data Analysis and Visualisations

# In[19]:


cacelled_perc = df['is_canceled'].value_counts(normalize = True)
print(cacelled_perc)

plt.figure(figsize = (5,4))
plt.title('Reservation status count')
plt.bar(['Not canceled', 'canceled'],df['is_canceled'].value_counts(), edgecolor = 'k', width = 0.7)
plt.show()


# In[20]:


plt.figure(figsize = (8,4))
ax1= sns.countplot(x= 'hotel', hue = 'is_canceled', data = df, palette = 'Blues')
legend_labels,_ = ax1. get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1,1))
plt.title('Reservation status in different hotels', size = 20)
plt.xlabel('hotel')
plt.ylabel('Number of reservation')
plt.show()


# In[21]:


resort_hotel = df[df['hotel'] == 'Resort Hotel']
resort_hotel['is_canceled'].value_counts(normalize = True)


# In[22]:


City_hotel = df[df['hotel'] == 'City Hotel']
City_hotel['is_canceled'].value_counts(normalize = True)


# In[23]:


resort_hotel = resort_hotel.groupby('reservation_status_date')[['adr']].mean()
city_hotel = City_hotel.groupby('reservation_status_date')[['adr']].mean()


# In[24]:


plt.figure(figsize =(20,8))
plt.title('Average Daily Rate in city and Resort Hotel', fontsize = 30)
plt.plot(resort_hotel.index, resort_hotel['adr'], label = 'Resort Hotel')
plt.plot(city_hotel.index, city_hotel['adr'], label = 'city Hotel')
plt.legend(fontsize = 20)
plt.show()


# In[25]:


df['month'] = df['reservation_status_date'].dt.month
plt.figure(figsize = (16,8))
ax1 = sns.countplot(x = 'month', hue = 'is_canceled', data = df, palette = 'bright')
legend_labels,_ = ax1.  get_legend_handles_labels()
ax1.legend(bbox_to_anchor=(1,1))
plt.title('Reservation status per month', size = 20)
plt.xlabel('month')
plt.ylabel('number of reservations')
plt.legend(['not canceled', 'canceled'])
plt.show()


# In[26]:


plt.figure(figsize = (15,18))
plt.title('ADR per month', fontsize = 30)
sns.barplot('month', 'adr', data = df[df['is_canceled'] ==1].groupby('month')[['adr']].sum().reset_index())
plt.show()


# In[27]:


canceled_data = df[df['is_canceled'] ==1]
top_10_country = canceled_data['country'].value_counts()[:10]
plt.figure(figsize = (8,8))
plt.title('Top 10 country with reservation canceled')
plt.pie(top_10_country, autopct = '%.2f', labels = top_10_country.index)
plt.show()


# In[28]:


df['market_segment'].value_counts()


# In[29]:


df['market_segment'].value_counts(normalize = True)


# In[30]:


canceled_data['market_segment'].value_counts(normalize = True)


# In[31]:


canceled_df_adr = canceled_data.groupby('reservation_status_date')[['adr']].mean()
canceled_df_adr.reset_index(inplace = True)
canceled_df_adr.sort_values('reservation_status_date', inplace = True)

not_canceled_data = df[df['is_canceled'] == 0]
not_canceled_df_adr  = not_canceled_data.groupby('reservation_status_date')[['adr']].mean()
not_canceled_df_adr.reset_index(inplace = True)
not_canceled_df_adr.sort_values('reservation_status_date', inplace = True)

plt.figure(figsize = (20,6))
plt.title('Average Daily Rate')
plt.plot(not_canceled_df_adr['reservation_status_date'],not_canceled_df_adr['adr'], label = 'not canceled')
plt.plot(canceled_df_adr['reservation_status_date'],canceled_df_adr['adr'], label = 'canceled')
plt.legend()


# In[32]:


canceled_df_adr = canceled_df_adr[(canceled_df_adr['reservation_status_date']>'2016') & (canceled_df_adr['reservation_status_date']<'2017-09')]
not_canceled_df_adr = not_canceled_df_adr[(not_canceled_df_adr['reservation_status_date']>'2016') & (not_canceled_df_adr['reservation_status_date']<'2017-09')]


# In[33]:


plt.figure(figsize = (20,6))
plt.title('Average Daily Rate')
plt.plot(not_canceled_df_adr['reservation_status_date'],not_canceled_df_adr['adr'], label = 'not canceled')
plt.plot(canceled_df_adr['reservation_status_date'],canceled_df_adr['adr'], label = 'canceled')
plt.legend(fontsize = 20)


# In[36]:


import calendar


# In[37]:


print(calendar.calendar(2023))


# In[39]:


print(calendar.month(2023,10))


# In[ ]:




