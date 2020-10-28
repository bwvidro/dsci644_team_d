# The script MUST include the following function,
# which is the entry point for this module:
# Param<dataframe1>: a pandas.DataFrame
# Param<dataframe2>: a pandas.DataFrame
# import statements
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import scipy
import pandas as pd

def azureml_main(dataframe1 = None, dataframe2 = None):
    dfCleaned = dataframe1
    dfReviews = dfCleaned['reviewText']
    dfTrain, dfTest = train_test_split(dfReviews, test_size=.1)
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
    # train_tfidf.todense()
    x_Values = pd.DataFrame(data=train_tfidf.todense())
    all_data = pd.merge(x_Values, dfCleaned[['reviewerRating']], left_index=True, right_index=True)
    return all_data
