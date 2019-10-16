# Udacity Data Scientist Project: Write A Data Science Blog Post

## Introduction
In this project, I choose to do analysis on UK companies gender pay gap data to quantify and measure the gender pay gap in UK.

## Project Overview
In order to tackle gender pay gap in UK, sicne 2017 UK Government asks all companies with more than 250 employees to publish Mean and Median Salary/Bonus differences between men and women. A positive value means men earn more than women. 

 - I looked at this data from various aspects and did some feature engineering to create new columns. 
 - I merged this dataset with SIC Codes dataset 
 - Using `pygeocode` we were able to add `state_code` parameter.
 - This analysis also shows that how difficult it is for women to climb up in salary ladder in general. 
 - I created a classifier using `DecisionTreeClassifier` from scikit-learn packet. Idea was to predict percentage of women in the Top Earners bucket for a new company. Based on publicly availabe data for a given company this model predicts modestly accurate if the percentage of women is "High", "Middle" or "Low" in Top Quartile.   

## Jupyter Notebook & Blog Post

Each step of analysis and model muilding is explained in `uk_gender_pay_gap.ipynb` Jupyter notebook. `uk_gender_pay_gap.ipynb` is in this directory.

I added a blog post on this analysis to [Medium](https://medium.com/@oneryalcin/please-mind-the-gender-gap-a7bc37bff781?sk=ec8a7514c0ed19ab20e351fa9b04aa6d), where I have extended comments for each step. 

*Mehmet Oner Yalcin*