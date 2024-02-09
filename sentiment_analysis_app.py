import streamlit as st
from transformers import pipeline

# Initialize the Hugging Face sentiment-analysis pipeline
# Note: In a real-world scenario, replace this with a call to the GPT API for sentiment analysis
sentiment_pipeline = pipeline("sentiment-analysis")

def convert_to_SSAT(score):
    """
    Converts a sentiment analysis score to a SSAT score between 1 and 5.
    This is a simplified example that assumes the input score is the positive
    sentiment probability.
    """
    if score <= 0.2:
        return 1
    elif score <= 0.4:
        return 2
    elif score <= 0.6:
        return 3
    elif score <= 0.8:
        return 4
    else:
        return 5

def analyze_sentiment(sentence):
    """
    Analyzes the sentiment of a sentence and returns an SSAT score.
    """
    results = sentiment_pipeline(sentence)
    # Assuming the result includes a positive sentiment score
    pos_score = results[0]['score'] if results[0]['label'] == 'POSITIVE' else 1 - results[0]['score']
    ssat_score = convert_to_SSAT(pos_score)
    return ssat_score

# Streamlit UI
st.title("Sentiment Analysis to SSAT Score Converter")

user_input = st.text_input("Enter a sentence for sentiment analysis:")

if user_input:
    ssat_score = analyze_sentiment(user_input)
    st.write(f"SSAT Score (1-5): {ssat_score}")
