# The script MUST contain a function named azureml_main
# which is the entry point for this module.

# imports up here can be used to 
import pandas as pd

# The entry point function can contain up to two input arguments:
#   Param<dataframe1>: a pandas.DataFrame
#   Param<dataframe2>: a pandas.DataFrame
def azureml_main(df1 = None, dataframe2 = None):

    posOrNeg = df1['reviewerRating'].apply(lambda x: 0 if x<.5 else 1)
    positiveNeutralNegative = df1['reviewerRating'].apply(lambda x: 0 if x<.6 else (1 if x<1 else (2)))
    reviewsScaledTo1To5 = df1['reviewerRating'].apply(lambda x: int(x*10/2))
            
    df1.insert(6, "posOrNeg", posOrNeg)

    df1.insert(7, "positiveNeutralNegative", positiveNeutralNegative)
    df1.insert(8, "reviewsScaledTo1To5", reviewsScaledTo1To5)

    return df1,
