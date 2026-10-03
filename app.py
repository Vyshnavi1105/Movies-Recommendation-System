import streamlit as st
import nltk
import sklearn
import pandas as pd
import pickle
import joblib

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.title("Movie Recommendation System")


with open("movies.pickle", 'rb') as m:
    movies = pickle.load(m)


# Create vectors and similarity
cv = CountVectorizer(max_features=5000, stop_words='english')

vectors = cv.fit_transform(movies['tags']).toarray()

similarity = cosine_similarity(vectors)


movie_names = movies['title'].values


def recommend(name_movie):

    movie_index = movies[movies['title'] == name_movie].index[0]

    recommendations = similarity[movie_index]

    movie_list = sorted(
        enumerate(recommendations),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    recommended_movies = []

    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)

    return recommended_movies


name_movie = st.selectbox("Enter the Movie Name", movie_names)


if st.button("Recommend"):

    r = recommend(name_movie)

    st.subheader("The Recommended Movies are:")

    for i in r:
        st.write(i)