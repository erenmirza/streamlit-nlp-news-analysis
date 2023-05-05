import requests
from textblob import TextBlob
import pandas as pd
from bs4 import BeautifulSoup
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import streamlit as st

def get_sentiment(text):
    blob = TextBlob(text)
    return blob.sentiment.polarity

def get_sentiment_polarity_category(polarity):
    if polarity > -0.33 and polarity < 0.33:
        return "Neutral"
    elif polarity <= -0.33:
        return "Negative"
    else:
        return "Positive"

def get_subjectivity(text):
    blob = TextBlob(text)
    return blob.sentiment.subjectivity

def get_subjectivity_category(polarity):
    if polarity >= 0.75:
        return "Very Factual"
    elif polarity >= 0.5 and polarity < 0.75:
        return "Somewhat Factual"
    elif polarity >= 0.25 and polarity < 0.5:
        return "Somewhat Subjective"
    else:
        return "Very Subjective"

def get_vader_sentiment(text):
    sentiment = SentimentIntensityAnalyzer()
    sentiment_scores = sentiment.polarity_scores(text)
    return sentiment_scores['compound']

def make_clickable(link, text):
    # target _blank to open new window
    return f'<a target="_blank" href="{link}">{text}</a>'

def main():
    # Set up the News API endpoint and API key
    url = "https://newsapi.org/v2/top-headlines"
    api_key = st.secrets["API_KEY"]

    # Set up the query parameters for the BBC news
    query_params = {
        "sources": "bbc-news",
        "language": "en"
    }

    # Send a GET request to the News API
    response = requests.get(url, params=query_params, headers={"Authorization": api_key}).json()

    # Extract articles from response and normalize into a dataframe
    articles = response['articles']
    df = pd.json_normalize(articles)
    df['Headline'] = df.apply(lambda x: make_clickable(x['url'], x['title']),axis=1)
    
    # Add article text to the dataframe
    for i, row in df.iterrows():
        article_url = row['url']
        article_response = requests.get(article_url, headers={"Authorization": api_key})
        article_content = article_response.content
        article_soup = BeautifulSoup(article_content, 'html.parser')
        article_text = article_soup.find_all(attrs={'class': 'ssrcss-1q0x1qg-Paragraph eq5iqo00'})
        pattern = '<[^<]+?>'
        clean_text = ''
        for line in article_text:
            clean_text = clean_text + '. ' + re.sub(pattern, "", str(line))
        if clean_text == '':
            clean_text = row['title']
        df.at[i, 'text'] = clean_text

    # # Perform sentiment analysis on article text
    # df['text_blob_sentiment'] = df['text'].apply(get_sentiment)
    # df['text_blob_sentiment_category'] = df['text_blob_sentiment'].apply(get_sentiment_polarity_category)

    # Perform subjectivity analysis on article text
    df['subjectivity_score'] = df['text'].apply(get_subjectivity)
    df['subjectivity_category'] = df['subjectivity_score'].apply(get_subjectivity_category)
    # df['subjectivity_rank'] = df['text_blob_subjectivity'].rank(method='max')

    # Perform sentiment analysis on article text using vader
    df['sentiment_score'] = df['text'].apply(get_vader_sentiment)
    df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_polarity_category)
    # df['sentiment_rank'] = df['vader_sentiment'].rank(method='min')

    st.set_page_config(layout="wide")
    st.subheader('Sentiment & Subjectivity Analysis of BBC Top 10 Articles')
    st.write(df[['publishedAt', 'Headline', 'subjectivity_score', 'subjectivity_category', 'sentiment_score', 'sentiment_category']].to_html(escape=False, index=False), use_container_width=True, unsafe_allow_html=True)
    with st.container():
        col1, col2 = st.columns(2)
        with col1:
            st.subheader('Sentiment of articles')
            st.bar_chart(df['subjectivity_category'].value_counts())
        with col2:
            st.subheader('Subjectivity of articles')
            st.bar_chart(df['sentiment_category'].value_counts())
            
if __name__ == "__main__":
    main()