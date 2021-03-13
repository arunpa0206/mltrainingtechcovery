import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#load daaset
df = pd.read_csv("movie_dataset.csv")

#feature column
features = ['keywords','cast','genres','director']

#create a function for combining the values of these columns into a single string.
def combine_features(row):
    return row['keywords'] +" "+row['cast']+" "+row["genres"]+" "+row["director"]

#We will fill all the NaN values with blank string in the dataframe.
for feature in features:
    #filling all NaNs with blank
    df[feature] = df[feature].fillna('')
#applying combined_features() method over each rows of dataframe and storing the combined string in “combined_features” column
df["combined_features"] = df.apply(combine_features,axis=1)

 #creating new CountVectorizer() object
cv = CountVectorizer()
#feeding combined strings(movie contents) to CountVectorizer() object
count_matrix = cv.fit_transform(df["combined_features"])
#we need to obtain the cosine similarity matrix from the count matrix.
cosine_sim = cosine_similarity(count_matrix)

def get_title_from_index(index):
    return df[df.index == index]["title"].values[0]

def get_index_from_title(title):
    return df[df.title == title]["index"].values[0]
#accessing the row corresponding to given movie to find all the similarity scores for that movie and then enumerating over it
movie_user_likes = "Avatar"
movie_index = get_index_from_title(movie_user_likes)
similar_movies =  list(enumerate(cosine_sim[movie_index]))

sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

i=0
print("Top 5 similar movies to "+movie_user_likes+" are:\n")
for element in sorted_similar_movies:
    print(get_title_from_index(element[0]))
    i=i+1
    if i>=5:
        break
