#!/usr/bin/env python
# coding: utf-8

# ### General idea
# 1. Clean dataset => dfClean
# 2. Vectorize words => to probability density
# 3. Perform logistic regression on vectorized words 
#     of scales of reviews 0 (0,.1),1 (.2,.3) ,2 (.4,.5),3 (.6,.7) ,4 (.8,.9, 1) reviews

# In[1]:


# Read data set and stop words
import pandas as pd
import re 
import nltk
nltk.download('stopwords')
nltk.download('words')
rawData = pd.read_csv (r'.\AppReview.csv')
#df=df1.sample(n=1000)
len(rawData.index)
rawData.columns


# In[5]:


rawData_v1=rawData[rawData["reviewerName"]!="A Google User"]
rawData_v1["reviewerRating"]=rawData["reviewerRating"]*10/2
rawData_v1=rawData_v1.astype({"reviewerRating": int})
rawData_v1.head()


# In[ ]:


rawData_v1.groupby("reviewerRating").nunique()


# In[34]:


# frames = []
# for i in range(1,6,1):
#     frames.append(rawData_v1[rawData_v1["reviewerRating"]==i/10].sample(n=1000, replace=False))

# df=pd.concat(frames,ignore_index =True)
# len(df.index)
#df=rawData_v1.sample(n=5000)
df=rawData_v1
#df.head()
df.groupby("reviewerRating").size()


# In[35]:


#df.shape[0]


# In[36]:


# Corpus of stop words
from nltk.corpus import stopwords
from nltk.corpus import words


# In[37]:


# processes a review and returns a list of words
def review_to_words(review, string = True, remove_stopwords=True):
    # Remove HTML
    #review_text = BeautifulSoup(review).get_text()
    review_text=review
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # Convert words to lower case and split them
    thesewords = review_text.lower().split()
    # Ignore non=english words
#     englishWords = words.words()
#     thesewords = [w for w in thesewords if w in englishWords]

    
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        thesewords = [w for w in thesewords if not w in stops]
    if string:
        return " ".join(thesewords)
    else:
        return thesewords


# In[38]:


#Clean up text
#Remove non-ascii text
#Remove all rows missing reviewerName
def fixString(x):
    return x.encode('ascii',errors='ignore')

# df[["reviewText"]]=df[["reviewText"]].apply(lambda x: str(x["reviewText"]).encode('ascii',errors='ignore').decode(), axis=1)
# df[["reviewerName"]]=df[["reviewerName"]].apply(lambda x: str(x["reviewerName"]).encode('ascii',errors='ignore').decode(), axis=1)
df[["reviewText"]]=df[["reviewText"]].apply(lambda x: review_to_words(x["reviewText"]), axis=1)
df[["reviewerName"]]=df[["reviewerName"]].apply(lambda x: str(x["reviewerName"]).encode('ascii',errors='ignore').decode(), axis=1)

dfCleaned=df[df['reviewText'].str.strip().astype(bool)]
dfCleaned=dfCleaned[df['reviewerName'].str.strip().astype(bool)]

#
#dfCleaned


# In[39]:


storedDFCleaned="dfCleaned_raw"
dfCleaned.to_pickle(storedDFCleaned)
#dfCleaned=pd.read_pickle(storedDFCleaned)


# ## Vectorize words
# ### This is based on 
# https://towardsdatascience.com/latent-semantic-analysis-sentiment-classification-with-python-5f657346f6a3
# 

# In[40]:


# import statements
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
tfidf.fit(dfCleaned["reviewText"])

X = tfidf.transform(dfCleaned["reviewText"])


# In[42]:


dfCleaned.iloc[1]["reviewText"]


# In[46]:


print([X[1, tfidf.vocabulary_['awful']]])


# ### Modelling part
# 1. Leverage the raw vector count and the tf-idf weighted version

# In[47]:


df=dfCleaned


# In[48]:


import numpy as np
df[df['reviewerRating'] != 3]
df['Positivity'] = np.where(df['reviewerRating'] > 3, 1, 0)
cols = ['appID', 'reviewerName', 'reviewerRating', 'reviewDate', 'textAnalytics']
df.drop(cols, axis=1, inplace=True)
df.dropna(inplace=True)
df.head()


# In[49]:


from sklearn.model_selection import train_test_split
X = df.reviewText
y = df.Positivity
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
print("Train set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_train),
                                                                             (len(X_train[y_train == 0]) / (len(X_train)*1.))*100,
                                                                            (len(X_train[y_train == 1]) / (len(X_train)*1.))*100))


# In[50]:


print("Test set has total {0} entries with {1:.2f}% negative, {2:.2f}% positive".format(len(X_test),
                                                                             (len(X_test[y_test == 0]) / (len(X_test)*1.))*100,
                                                                            (len(X_test[y_test == 1]) / (len(X_test)*1.))*100))


# In[51]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
def accuracy_summary(pipeline, X_train, y_train, X_test, y_test):
    sentiment_fit = pipeline.fit(X_train, y_train)
    y_pred = sentiment_fit.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("accuracy score: {0:.2f}%".format(accuracy*100))
    return accuracy


# In[52]:


cv = CountVectorizer()
rf = RandomForestClassifier(class_weight="balanced")
n_features = np.arange(10000,30001,10000)
def nfeature_accuracy_checker(vectorizer=cv, n_features=n_features, stop_words=None, ngram_range=(1, 1), classifier=rf):
    result = []
    print(classifier)
    print("\n")
    for n in n_features:
        vectorizer.set_params(stop_words=stop_words, max_features=n, ngram_range=ngram_range)
        checker_pipeline = Pipeline([
            ('vectorizer', vectorizer),
            ('classifier', classifier)
        ])
        print("Test result for {} features".format(n))
        nfeature_accuracy = accuracy_summary(checker_pipeline, X_train, y_train, X_test, y_test)
        result.append((n,nfeature_accuracy))
    return result
tfidf = TfidfVectorizer()
print("Result for trigram with stop words (Tfidf)\n")
feature_result_tgt = nfeature_accuracy_checker(vectorizer=tfidf,ngram_range=(1, 3))


# In[53]:


from sklearn.metrics import classification_report
cv = CountVectorizer(max_features=30000,ngram_range=(1, 3))
pipeline = Pipeline([
        ('vectorizer', cv),
        ('classifier', rf)
    ])
sentiment_fit = pipeline.fit(X_train, y_train)
y_pred = sentiment_fit.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['negative','positive']))


# In[ ]:




