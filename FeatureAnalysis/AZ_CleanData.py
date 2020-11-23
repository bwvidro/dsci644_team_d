# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd
import re 
import nltk

# Corpus of stop words
from nltk.corpus import stopwords

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

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame

def azureml_main(dataframe1 = None, dataframe2 = None):

    # Execution logic goes here
    print('Input pandas.DataFrame #1:\r\n\r\n{0}'.format(dataframe1))
    # nltk.download('stopwords')
    df = dataframe1
    df[["reviewText"]]=df[["reviewText"]].apply(lambda x: review_to_words(x["reviewText"]), axis=1)
    df[["reviewerName"]]=df[["reviewerName"]].apply(lambda x: str(x["reviewerName"]).encode('ascii',errors='ignore').decode(), axis=1)

    dfCleaned=df[df['reviewText'].str.strip().astype(bool)]
    dfCleaned=dfCleaned[df['reviewerName'].str.strip().astype(bool)]

    dfCleaned

    # If a zip file is connected to the third input port is connected,
    # it is unzipped under ".\Script Bundle". This directory is added
    # to sys.path. Therefore, if your zip file contains a Python file
    # mymodule.py you can import it using:
    # import mymodule
    
    # Return value must be of a sequence of pandas.DataFrame
    return dfCleaned,
