# Udacity Data Scientist Project: Write A Data Science Blog Post

## Introduction
In this project, I choose to do analysis on UK companies gender pay gap data to quantify and measure the gender pay gap in UK using CRISP-DM method.

### Business Understanding
In order to understand Gender Pay Gap difference I asked a few questions:

- Does men earn more on average/median more than women?
- If so, how much do they earn more?
- Which sectors employ women more?
- Are these sectors give enough opportunities to women to get promoted and earn more?
- Which sectors are the most difficult ones for women to advance?
- There are number of companies which submitted their gender pay gap data yet. Is there a way to predict the gender pay gap in general? Can we build a statistical model?

### Data Understanding & Data Preperation
Data understanding was detailed more deep in the accompanying jupyter notebook, but essentially there are two datasets being used:
- UK Government Gender Pay Gap data
- SIC Codes from `datahub.io`
- Feature engineered country code from post code

Gender Pay Gap Data provides two statistics (mean and median) for Difference of Hourly Salary between men and women in each company, as well as percentage of women/men in four different salary buckets.

SIC codes are the area of opearion for a given company. They are encoded as numbers, but we get sector names as new columns when SIC codes dataset was joined with UK Govenment Gende Pay Gap data.

- In order to use in the statistical model we built at the end, we needed county codes in UK, like England, Wales, Scotland and Northern Ireland. Post codes are extracted from `Address` field and using `pygeocode` we mapped it to  `state_code` column.

### Evaluation
We answered questions (except the last one) introduced in **Business Understanding** section using the data in previous section. We evaluated the results and discussed the implications of the results. 

### Modelling
The last question was answered by building a Machine Learning (Classification) model using scikit-learn's `DecisionTree` Classifier. The accuracy of the model is discussed in the blog post. 

## Jupyter Notebook & Blog Post

Each step of analysis and model muilding is explained in `uk_gender_pay_gap.ipynb` Jupyter notebook. `uk_gender_pay_gap.ipynb` is in this directory.

I added a blog post on this analysis to [Medium](https://medium.com/@oneryalcin/please-mind-the-gender-pay-gap-9162f13b4202?sk=a98121fa202bd6851d99eb2f9438a246), where I have extended comments for each step. 

*Mehmet Oner Yalcin*