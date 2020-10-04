#!/usr/bin/env python
# coding: utf-8

# ### General idea
# 1. Clean dataset => dfClean
# 2. Vectorize words => to probability density
# 3. Perform logistic regression on vectorized words 
#     of scales of reviews 0 (0,.1),1 (.2,.3) ,2 (.4,.5),3 (.6,.7) ,4 (.8,.9, 1) reviews

# In[35]:


# Read data set and stop words
import pandas as pd
import re 
import nltk
nltk.download('stopwords')
df = pd.read_csv (r'.\AppReview.csv')
len(df.index)


# In[36]:


# Corpus of stop words
from nltk.corpus import stopwords


# In[37]:


# processes a review and returns a list of words
def review_to_words(review, string = True, remove_stopwords=True):
    # Remove HTML
    #review_text = BeautifulSoup(review).get_text()
    review_text=review
    # Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    # Convert words to lower case and split them
    words = review_text.lower().split()
    # Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    if string:
        return " ".join(words)
    else:
        return words


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
dfCleaned


# In[39]:


dfCleaned[['reviewText']]


# ## Vectorize words
# ### This is based on 
# https://towardsdatascience.com/sentiment-analysis-a-how-to-guide-with-movie-reviews-9ae335e6bcb2
# With actual logistic regression:
# https://towardsdatascience.com/sentiment-classification-with-logistic-regression-analyzing-yelp-reviews-3981678c3b44
# 

# In[40]:


from sklearn.model_selection import train_test_split

dfReviews = dfCleaned['reviewText']
dfTrain, dfTest = train_test_split(dfReviews, test_size=.1)


# In[41]:


# import statements
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

# Initialize a bag of words
#vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 5000) 
vectorizer = CountVectorizer(analyzer = "word", tokenizer = None, preprocessor = None, stop_words = None, max_features = 1000) 

# Fit transform the data 
train_feat = vectorizer.fit_transform(dfTrain).toarray()
test_feat = vectorizer.transform(dfTest).toarray()


# TFIDF train set
tfidf_transformer = TfidfTransformer().fit(train_feat)
train_tfidf = tfidf_transformer.transform(train_feat)
 
# apply tfidf to test set
test_tfidf = tfidf_transformer.transform(test_feat)


# In[42]:


#train data
trainYdata = pd.merge(dfTrain.to_frame(), dfCleaned[['reviewerRating']], left_index=True, right_index=True)
#determined feature names
featureNamesList=vectorizer.get_feature_names()


# In[43]:


# look at data for training
type(dfTrain)
stopcounter = 0
for index, value in dfTrain.items():
    print(f"Index : {index}, Value : {value}")
    stopcounter=stopcounter+1
    if stopcounter > 10:
        break


# In[44]:


# Dump some vectorized words and probablities
import scipy.sparse

featureNamesList=vectorizer.get_feature_names()
type(dfTrain)

cx = scipy.sparse.coo_matrix(train_feat)
cx2 = scipy.sparse.coo_matrix(train_tfidf)


print(dfTrain.iloc[0])
for i,j,v in zip(cx.row, cx.col, cx.data):
    if i!=0:
        break
    print("(%d, %d), %s = %s" % (i,j,featureNamesList[j], v))
    
print("\r\n\r\nThe weighted results\r\n")
for i,j,v in zip(cx2.row, cx2.col, cx2.data):
    if i!=0:
        break
    print("(%d, %d), %s = %s" % (i,j,featureNamesList[j], v))


# In[45]:


# look at some more data
import scipy.sparse

featureNamesList=vectorizer.get_feature_names()
type(test_tfidf)

cx = scipy.sparse.coo_matrix(test_tfidf)

print(dfTest.iloc[0])
for i,j,v in zip(cx.row, cx.col, cx.data):
    if i!=0:
        break
    print("(%d, %d), %s = %s" % (i,j,featureNamesList[j], v))


# ### Modelling part
# 1. Leverage the raw vector count and the tf-idf weighted version

# In[46]:


# # Just looking at some data
# train_tfidf
# dfTrain
# print(df.iloc[105146])
# dfTrain
# df
# train_tfidf.todense()
# dfTrain.to_frame()


# In[47]:


# Get trained Y data and test Y data
trainYdata = pd.merge(dfTrain.to_frame(), dfCleaned[['reviewerRating']], left_index=True, right_index=True)
testYdata = pd.merge(dfTest.to_frame(), dfCleaned[['reviewerRating']], left_index=True, right_index=True)


# In[48]:


# Get trained X data and test X data
trainXdata = train_tfidf.todense()
testXdata = test_tfidf.todense()


# In[49]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression


# In[53]:


# Train data - map the y to ints of scales of reviews 0,1,2,3,4 reviews
y = trainYdata[['reviewerRating']]
y_int = trainYdata['reviewerRating'].apply(lambda x: 0 if x<.2 else (1 if x<.4 else (2 if x<.6 else (3 if x<.8 else 4))))

X = trainXdata

clf = LogisticRegression(random_state=0, max_iter=1000).fit(X, y_int)

# Check trained accuracy
clf.predict(X[:2, :])
clf.predict_proba(X[:2, :])
clf.score(X, y_int)


# ### Check test accuracy

# In[54]:


y_int_test = testYdata['reviewerRating'].apply(lambda x: 0 if x<.2 else (1 if x<.4 else (2 if x<.6 else (3 if x<.8 else 4))))


# In[55]:


clf.predict(testXdata)
clf.predict_proba(testXdata)
clf.score(testXdata, y_int_test)


# In[ ]:




