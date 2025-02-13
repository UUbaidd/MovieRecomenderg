#!/usr/bin/env python
# coding: utf-8

# In[1]:


#content base recommender system
import numpy as np
import pandas as pd


# In[2]:


credits = pd.read_csv(r"D:\kaggle\Movie Recommender\tmdb_5000_credits.csv")
movies = pd.read_csv(r"D:\kaggle\Movie Recommender\tmdb_5000_movies.csv")


# In[3]:


movies.head()


# In[4]:


credits.head()


# In[5]:


#merging data
movies= movies.merge(credits,on='title')


# In[6]:


movies.head()


# In[7]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[8]:


movies.head()


# In[9]:


movies.isnull().sum()


# In[10]:


movies.dropna(inplace=True)


# In[11]:


movies.isnull().sum()


# In[12]:


movies.duplicated().sum()


# In[13]:


movies.iloc[0].genres


# In[14]:


import ast
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[15]:


movies['genres']=movies['genres'].apply(convert)


# In[16]:


movies['genres']=movies['keywords'].apply(convert)


# In[17]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[18]:


def convert3(obj):
    L = []
    counter=0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter+=1
        else:
            break
    return L


# In[19]:


movies['cast']=movies.cast.apply(convert3)


# In[20]:


movies.head()


# In[21]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[22]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[23]:


movies.head()


# In[24]:


movies['overview'][0]


# In[25]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[26]:


movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[27]:


movies.head()


# In[28]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[29]:


movies.head()


# In[30]:


new_df = movies[['movie_id','title','tags']]


# In[31]:


new_df


# In[32]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[33]:


new_df.head()


# In[34]:


import nltk


# In[35]:


from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()


# In[36]:


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# In[37]:


new_df.tags.apply(stem)


# In[38]:


new_df['tags'][0]


# In[39]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[40]:


new_df.head()


# In[41]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words='english')


# In[42]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[43]:


vectors


# In[44]:


cv.get_feature_names_out()


# In[45]:


#steming


# In[49]:


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vectors)


# In[50]:


similarity


# In[51]:


sorted(list(enumerate(similarity[0])),reverse=True, key= lambda x:x[1])[1:6]


# In[55]:


def recommend(movie):
    movie_index = new_df[new_df['title']==movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True, key= lambda x:x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)
        


# In[56]:


#index
recommend('Avatar')


# In[ ]:




