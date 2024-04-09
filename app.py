import streamlit as st
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter

# Download VADER lexicon
nltk.download('vader_lexicon')

# Function to analyze sentiment
def analyze_sentiment(text):
    sid = SentimentIntensityAnalyzer()
    score = sid.polarity_scores(text)['compound']
    if score > 0:
        return 'Positive'
    elif score == 0:
        return 'Neutral'
    else:
        return 'Negative'

# Function to extract most frequent positive and negative words
def extract_sentiment_words(text):
    sid = SentimentIntensityAnalyzer()
    words = text.split()
    positive_words = [word for word in words if sid.polarity_scores(word)['compound'] > 0]
    negative_words = [word for word in words if sid.polarity_scores(word)['compound'] < 0]
    return positive_words, negative_words

# Function to generate word cloud
def generate_wordcloud(words, title):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(Counter(words))
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(title)
    plt.axis('off')
    st.pyplot(plt)

# Streamlit app layout
def main():
    st.title('Sentiment Analysis with Word Cloud and Pie Chart')
    st.write('Enter a paragraph and I will analyze its sentiment.')

    # Input text box
    paragraph = st.text_area('Enter a paragraph:', height=200)

    # Perform sentiment analysis
    if st.button('Analyze Sentiment'):
        if paragraph.strip() != '':
            sentiment = analyze_sentiment(paragraph)
            st.write(f'The sentiment of the paragraph is: {sentiment}')

            # Extract positive and negative words
            positive_words, negative_words = extract_sentiment_words(paragraph)

            # Generate and display word clouds for positive and negative sentiment words
            generate_wordcloud(positive_words, 'Positive Sentiment Words')
            generate_wordcloud(negative_words, 'Negative Sentiment Words')

            # Calculate sentiment percentages
            sid = SentimentIntensityAnalyzer()
            scores = sid.polarity_scores(paragraph)
            positive_percentage = scores['pos'] * 100
            negative_percentage = scores['neg'] * 100

            # Plot pie chart for sentiment percentages
            labels = ['Positive', 'Negative']
            sizes = [positive_percentage, negative_percentage]
            colors = ['#66b3ff', '#ff9999']
            explode = (0.1, 0)  # explode 1st slice
            plt.figure(figsize=(6, 6))
            plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Sentiment Percentage')
            st.pyplot(plt)
        else:
            st.write('Please enter a paragraph.')

if __name__ == "__main__":
    main()
