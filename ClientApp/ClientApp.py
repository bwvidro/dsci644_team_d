import urllib
# If you are using Python 3+, import urllib instead of urllib2
import requests
import json 
import pandas
import datetime
import random
import argparse
#import numpy as np
import datetime
import os
import re
import time
import math
from mpi4py import MPI

mpi_comm  = MPI.COMM_WORLD
status    = MPI.Status()
comm_size = mpi_comm.Get_size()
rank      = mpi_comm.Get_rank()

df = pandas.read_csv('ClientTest.csv')
sampleSize=10
started = datetime.datetime.now()
sampled_range = random.sample(range(len(df.index)), sampleSize)
#print("Expected Actual")
success = 0;
for i in sampled_range:
    reviewText = df.loc[i][0];
    expected = df.loc[i][1];
    data =  {

            "Inputs": {

                    "input1":
                    {
                        "ColumnNames": ["reviewText"],
                        "Values": [ [ reviewText ] ]
                    },        },
                "GlobalParameters": {
            }
        }

    
    initBody = json.dumps(data)
    body = str.encode(initBody)

    url = 'https://ussouthcentral.services.azureml.net/workspaces/7654e77b4a8c4df19d3774dad4e5bd99/services/e984c0fdf53b4379880a967ac949fd7e/execute?api-version=2.0&details=true'
    api_key = 'TN59tYi2fz/rI93YwmiTRVjThRGBGjQ+vnL+Gf3sKlv34kmvLzPske0SsR23bRiOQHpyMdgWKvUNaWTHwvTClA==' # Replace this with the API key for the web service
    headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ api_key)}

    response = requests.post(url, headers=headers, data=body) 
    if (response.json()["Results"]["output1"]["value"]["Values"][0][0] == "Positive"):
        actual=1
    else:
        actual=0
    if expected==actual:
        success=success+1
    
    #print(expected, actual)
    
completed = datetime.datetime.now()
averageTime = completed-started
print("Process:", rank, " ", (averageTime/sampleSize).total_seconds(), "s ", success/sampleSize, "%", sep="")