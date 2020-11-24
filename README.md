# **Software Engineering for Data Science (DSCI-644) Project in Microsoft Azure**

## **Project Objective** 
The goal of this project is to re-design and re-train the model to give better performance.

We developed the model using Microsoft Azure ML Studio and tracked our overall progress of the project using Trello.  Other technologies utilized include PyCharm, Jupyter Notebooks, Python (Execute Custom Python Scripts inside of Azure), Git

Trello board: https://trello.com/b/4XceHD7e/term-project
Git Repository: https://github.com/bwvidro/dsci644_team_d 
Webpage: https://bwvidro.github.io/dsci644_team_d/

The method we chose:
    1) Grouped reviews into 2 categories (Positive and Negative) - any values below .5 were assumed Negative and any values about .5 were assumed Positive
    2) Clean Data 
        a) Remove Stop Words 
    3) The final application allows a user to enter a review as user input and the model will predict whether that review is Positive or Negative.

### Model Investigations:

1) Cleaning data
    a) Removed stop words
    b) Cleaned out non-UTF8 encoded data
        i) **This had an adverse effect on the outcome, and not implemented**
    c) Remove non-letters
        i) review_text = re.sub("[^a-zA-Z]"," ", review_text)
    d) Convert words to lower-case and split them
    e) Filter out non-English words
        i) **This had an adverse effect on the outcome, and not implemented**
2) Sampling data
    a) After reviewing the data and noticing it had an unbalanced classification. The distribution was negatively skewed meaning most of the reviews were positive (skewed to the right).
    b) We adjusted sampling to ensure all classes were equally represented, using sampling with a replacement if necessary.
        i) **This turned out to be better handled in the modeling tool selected instead of adjusting the initial sample**
3) Word vectorization
    a) Went from Latent Dirichlet Allocation to Tf-Idf vectorization using n-gram features from words
        i) Tf-Idf tried various ranges, 1000 words seemed to be enough
4) Modeling
    a) Attempted the following:
        i) Default neural network
        ii) Logistic regression
        iii) SVM
        iv) Random Forest
    b) **Logistic regression was selected based on best learning for the data**

**Model:** The initial model used a neural network that would give about 60% accuracy based on the reviews. The dataset was split and 75% of the dataset was used for training and 25% was used for testing. 

### Model Training Structure

![Train Model](https://github.com/bwvidro/dsci644_team_d/blob/master/Architecture/Model_Train.JPG)

### Model Predictive Structure

![Prediction Model](https://github.com/bwvidro/dsci644_team_d/blob/master/Architecture/Model_Pred.JPG)

### Configuration Management

1. Initial Research and development
    * Performed on various individual developers machines
        i Pycharm
        ii Jupyter Notebook
    * Code versioning tracked and shared via git
    
2. Initial implementation of solutions
    * Azure ML
        i Each developer leveraged their own workspace for prototyping
        ii Sharing of work done via experiments and coping experiments to each other’s workspaces

    * Versioning is done via naming conventions

### Result

![Results](https://github.com/bwvidro/dsci644_team_d/blob/master/Results/Results.JPG)

![Score](https://github.com/bwvidro/dsci644_team_d/blob/master/Results/Result.JPG)

### Web Service API

Simple API for prediction, the input is review text only, the output is a prediction of review (Model in **Model Predictive Structure**)

#### Usage

Enter review in the REVIEWTEXT Box

![API](https://github.com/bwvidro/dsci644_team_d/blob/master/Results/API.JPG)

Output:

Details of response API in Json - showing 67% change of the review being negative

![Output](https://github.com/bwvidro/dsci644_team_d/blob/master/Results/Output.JPG)

## Client API

The API was implemented in Python that is able to submit text, check response and get the performance for a sample size of 50.

## GIT Repository:

**Architecture** contains the proposed model for the client in Microsoft Azure

**ClientAPP** contains a Python implementation for an application that allows a user to enter a review as user input to the azure model which will predict whether that review is Positive or Negative.

**Feature Analysis** contains different models implemented on Python

**Presentation** contains the project presentation

**Results** contains the jpg of the results obtained by the model as well the web API output

**instructions** contains the project objective and requirements.

