import os
os.getcwd()
os.chdir("C:/Users/namra/Desktop/5TH SEM/PYTHON ML_LAB")
os.getcwd()
import pandas as pd
data1 = pd.read_csv("customer_reviews.csv")

import nltk
nltk.download("vader_lexicon")
from nltk.sentiment.vader import SentimentIntensityAnalyzer
senti_analysis=SentimentIntensityAnalyzer()
senti_analysis.polarity_scores(data1.iloc[23,1])

print(data1.iloc[23,1])
data1["score"]=data1["text"].apply(lambda x:senti_analysis.polarity_scores(x))
data1["compound_score"]=data1["score"].apply(lambda x:x["compound"])

import numpy as np
data1["positive_negative"]=data1["compound_score"].apply(lambda x:np.where(x>0,"positive","negative"))
data1["positive_negative"].value_counts()
positive_data=data1.query("positive_negative=='positive'")
negative_data=data1.query("positive_negative=='negative'")
