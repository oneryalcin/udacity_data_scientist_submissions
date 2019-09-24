# Udacity Data Scientist Capstone Project: Predicting Customer Churn

## Introduction
As the capstone project I decided to go for working on Big Data and choose predicting customer churn using Apache Spark.

In this project I used `pyspark` to work on Sparkify music streaming user activity data. Few major steps are:
 - Data ingesting and cleaning
 - Handling missing data
 - Exploratory analyisis
 - Feature engineering
 - Model building and testing few models
 - Adding evaluators to measure success
 - Optimising hyperprarmeters with Grid Search
 - Using Cross Validation to ensure best use of training data
 - Refactor all code using Pipelines

## Project Overview
At Sparkify we want to predict user churn so that we identify users to give promotion in order to keep user engagement as high as possible. We have a 128MB of user activity session data where each rows explains a user activity such as listening to a song, upgrading/downgrading susbcription, adding a new song to playlist. Originally data is 12GB which is much bigger than our workstation memory. Therefore we use Apache Spark to benefit from parallel and scalable data analysis, model building and evaluation.

By applying individual steps outlined in `Introduction` section we built a relatively successful classification model, that predicts like to churn users.

## Jupyter Notebook & Blog Post

Each step of building a customer churn model is in detail explained in the `Sparkify.ipynb` Jupyter notebook. `Sparkify.ipynb` is in this directory.

I would highly suggest you to read [my Medium Blog post](https://medium.com/@oneryalcin/finding-needle-in-haystack-with-apache-spark-eb4c846f998d), where I have extended comments for each step. Please [click here](https://medium.com/@oneryalcin/finding-needle-in-haystack-with-apache-spark-eb4c846f998d) for the Medium blog post.

*Mehmet Oner Yalcin*
