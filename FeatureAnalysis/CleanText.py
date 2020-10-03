#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

df = pd.read_csv (r'.\AppReview.csv')
len(df.index)


# In[2]:


#Remove non-ascii text
#Remove all rows missing reviewerName
def fixString(x):
    return x.encode('ascii',errors='ignore')

df[["reviewText"]]=df[["reviewText"]].apply(lambda x: str(x["reviewText"]).encode('ascii',errors='ignore').decode(), axis=1)
df[["reviewerName"]]=df[["reviewerName"]].apply(lambda x: str(x["reviewerName"]).encode('ascii',errors='ignore').decode(), axis=1)

dfCleaned=df[df['reviewText'].str.strip().astype(bool)]
dfCleaned=dfCleaned[df['reviewerName'].str.strip().astype(bool)]

len(dfCleaned.index)


# In[3]:


dfCleaned.head(10)


# ### This is based on 
# https://towardsdatascience.com/sentiment-analysis-a-how-to-guide-with-movie-reviews-9ae335e6bcb2

# In[4]:


from sklearn.model_selection import train_test_split

dfReviews = df['reviewText']
dfTrain, dfTest = train_test_split(dfReviews, test_size=.1)


# In[5]:


# import statements
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Initialize a bag of words
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
# Fit transform the data 
train_feat = vectorizer.fit_transform(dfTrain).toarray()
test_feat = vectorizer.transform(dfTest).toarray()


# TFIDF train set
tfidf_transformer = TfidfTransformer().fit(train_feat)
train_tfidf = tfidf_transformer.transform(train_feat)
 
# apply tfidf to test set
test_tfidf = tfidf_transformer.transform(test_feat)


# In[26]:


vectorizer


# In[25]:


x = train_feat[0].tolist()
len(x)


# In[30]:


dfTrain[1]


# In[28]:



for index in range(len(x)):
    if (x[index]==1):
        print(featureNamesList[index])


# In[6]:


import numpy
numpy.version.version


# In[27]:


#type(test_tfidf)
featureNamesList=vectorizer.get_feature_names()
print(featureNamesList[2904])
#print(test_tfidf)
#print(test_tfidf.multiply(test_tfidf >= 0.9))


# In[8]:


train_tfidf.toarray()


# In[8]:


tfidf_transformer.get_params()


# In[9]:


df.apply (lambda row: featureNamesList(row), axis=1)


# In[ ]:




