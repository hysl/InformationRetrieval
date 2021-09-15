import wget

import os
import json
import gzip
import pandas as pd
from urllib.request import urlopen


import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import math
import time
from sklearn.metrics.pairwise import cosine_similarity

import warnings; warnings.simplefilter('ignore')


from tkinter import *

#download All Beauty metadata
wget.download('http://deepyeti.ucsd.edu/jianmo/amazon/metaFiles/meta_All_Beauty.json.gz')

#open All Beauty metadata json file
data = []
with gzip.open('meta_All_Beauty.json.gz') as f:
    for l in f:
        data.append(json.loads(l.strip()))

# convert list into pandas dataframe
df = pd.DataFrame.from_dict(data)

#convert dataframe to csv file
df.to_csv('meta_all_beauty.csv', sep=',', encoding='utf-8')

#read csv file
meta_data = pd.read_csv("meta_all_beauty.csv")

#drop unnecessary features
meta_data.drop(labels = ['Unnamed: 0', 'category', 'tech1', 'fit', 'also_buy', 'image', 'tech2', 'brand', 'feature', 'rank', 'also_view', 'main_cat', 'similar_item', 'date', 'price', 'details'], axis=1)

#drop rows where description is empty
meta_data_cleaned = meta_data.copy().loc[meta_data['description'] != '[]']




#download All Beauty reviews
wget.download('http://deepyeti.ucsd.edu/jianmo/amazon/categoryFiles/All_Beauty.json.gz')

#open All Beauty reviews json file
data1 = []
with gzip.open('All_Beauty.json.gz') as f:
    for l in f:
        data1.append(json.loads(l.strip()))

# convert list into pandas dataframe
df1 = pd.DataFrame.from_dict(data1)

#convert dataframe to csv file
df1.to_csv('ratings_all_beauty.csv', sep=',', encoding='utf-8')

#read csv file
ratings_data = pd.read_csv("ratings_all_beauty.csv")


#drop unnecessary features
ratings_data.drop(labels = ['Unnamed: 0', 'verified','reviewerName', 'reviewText', 'summary', 'unixReviewTime', 'style', 'image'], axis=1)

#rename overall to ratings
ratings_data.rename(columns={'overall': 'ratings'}, inplace=True)

#format vote values
ratings_data['vote'] = ratings_data['vote'].str.replace(',', '')
ratings_data['vote'] = ratings_data['vote'].fillna(0)
ratings_data['vote'] = ratings_data['vote'].astype(float)

#format date
ratings_data['reviewTime'] = pd.to_datetime(ratings_data['reviewTime'], format='%m %d, %Y')



new_ratings_data = ratings_data
new_ratings_data.drop(labels = 'reviewerID', axis=1)

#new_ratings_data.sort_values(by=['asin', 'reviewTime', 'ratings', 'vote'], inplace=True, ascending=True)

#find weighted average
new_ratings_data["total_ratings"] = ''
new_ratings_data["total_votes"] = ''
new_ratings_data["total_ratings"] = new_ratings_data.groupby(['asin']).transform('count')
new_ratings_data["total_votes"] = new_ratings_data.groupby(['asin'])['vote'].transform('sum')

ratings_votes_data = new_ratings_data.groupby(['asin']).agg({'ratings': 'mean', 'reviewTime': 'max', 'vote': 'mean', 'total_ratings': 'max', 'total_votes': 'max'})

a = ratings_votes_data['ratings']
b = ratings_votes_data['vote']
c = ratings_votes_data['total_ratings']
d = ratings_votes_data['total_votes']

ratings_votes_data['weighted_scores'] = ((b*c) + (a*d)) / (c+d)


#merge all data
all_data = pd.merge(meta_data_cleaned, ratings_votes_data, on='asin')

#training model with training set
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(stop_words='english')

#perform tfidf of description feature
meta_matrix = tfidf.fit_transform(all_data['description'])

#the linear kernel here is the same as the cosine similarity, but faster
cosine_similarity_all_beauty = linear_kernel(meta_matrix, meta_matrix)


product_title = pd.Series(all_data.index, index = all_data['title'])

#recommend function
def recommend(title_input):
  
  id = product_title[title_input]
  similarity_scores = list(enumerate(cosine_similarity_all_beauty[id]))
  similarity_scores = sorted(similarity_scores, key = lambda x: x[1], reverse=True)
  similarity_scores = similarity_scores[1:51]

  product_index = [i[0] for i in similarity_scores]

  #get ratings and time of similar products
  rating = all_data['weighted_scores'].iloc[product_index]
  time = all_data['reviewTime'].iloc[product_index]

  df1 = pd.DataFrame.from_dict(rating).reset_index()
  df1['index'] = df1['index'].astype(float)
  df1.columns = ['ind', 'weight_ratings']

  new_sim = np.array(similarity_scores)
  df2 = pd.DataFrame.from_dict(new_sim)
  df2.columns = ['ind', 'sim_scores']

  df3 = pd.DataFrame.from_dict(time).reset_index()
  df3['index'] = df3['index'].astype(float)
  df3.columns = ['ind', 'reviewTime']

  new_dataset = pd.merge(df1, df2, on='ind')
  new_dataset = pd.merge(new_dataset, df3, on='ind')

  a = new_dataset['weight_ratings']
  b = new_dataset['sim_scores']

  #find new score
  new_dataset['new_scores'] = (a*b)/2

  new_dataset_final = new_dataset.sort_values(['new_scores', 'reviewTime'], ascending=False).head(20)

  return all_data['title'].iloc[new_dataset_final['ind']].head(20)


#testing out Beauty Product Recommender with:
#"bareMinerals Prime Time BB Primer Cream Fair"
#and then also try
#"Bacid Probiotic with Bacillus Coagulans for Digestive Health, 100 Capsules"

#create the GUI
main = Tk()
main.title("Beauty Product Recommender")
main.geometry("500x500")

#recommender function
def recommender():
    temp = inputBox.get()
    output.delete('1.0', END)
    output.insert(END, recommend(temp))
    inputBox.delete(0, END)

#input box
inputBox = Entry(main, width=51, bg="light grey")
inputBox.grid(row=0, column=0, sticky=W)
inputBox.focus_set()

#recommend button
Button(main, width=20, text="Recommend", command = recommender).grid(row=1, column=0, sticky=W)

#output box
output = Text(main, width=70, height=50, background = "light grey")
output.grid(row=2, column=0, sticky=W)

main.mainloop()




