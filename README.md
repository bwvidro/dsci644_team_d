# **Software Engineering for Data Science (DSCI-644) Project in Microsoft Azure**

## **Project Objective** 
The goal of this project is to re-design and re-train the model to give better performance.

**Model:** The initial model used a neural network that would give about 60% accuracy based on the reviews. The dataset was split and 75% of the dataset was used for training and 25% was used for testing. 

### Model Training Structure

![Train Model](https://github.com/bwvidro/dsci644_team_d/blob/master/Architecture/Model_Train.JPG)

### Model Predictive Structure

![Prediction Model](https://github.com/bwvidro/dsci644_team_d/blob/master/Architecture/Model_Pred.JPG)

### Result

![Results](https://github.com/bwvidro/dsci644_team_d/blob/master/Results/Results.JPG)

![Score](https://github.com/bwvidro/dsci644_team_d/blob/master/Results/Result.JPG)

### Web Service API

Simple API for prediction, the input is review text only, the output is a prediction of review (Model in **Model Predictive Structure**)

#### Usage

Enter review in the '''REVIEWTEXT''' Box

![API](https://github.com/bwvidro/dsci644_team_d/blob/master/Results/API.JPG)

Output:
![Output](https://github.com/bwvidro/dsci644_team_d/blob/master/Results/Output.JPG)



**Architecture** contains the proposed model for the client in Microsoft Azure

**Feature Analysis** contains different models implemented on Python

**ClientApp** contains a Python implementation for an application that allows a user to enter a review as user input to the azure model which will predict whether that review is Positive or Negative.

