# -*- coding: utf-8 -*-
"""sentimental_model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1vXZAps_wV6PlxytiOmYgTbBY5soNb0Av
"""

import re
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline
from torch.nn.functional import softmax

# Load pre-trained BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name)

# Initialize Zero-Shot Classification pipeline
zero_shot_pipeline = pipeline("zero-shot-classification")

def preprocess_tweet(tweet):
    # Remove punctuation, symbols, @mentions, #hashtags, and links
    tweet = re.sub(r'[^\w\s#@]', '', tweet)
    tweet = re.sub(r'http\S+', '', tweet)
    return tweet.strip()

def combined_sentiment(bert_sentiment, zero_shot_sentiment, bert_prob, zero_shot_prob):
    if bert_sentiment == 'Positive' and zero_shot_sentiment == 'positive':
        return 'positive'
    elif bert_sentiment == 'Negative' and zero_shot_sentiment == 'negative':
        return 'negative'
    else:
        # Use a weighted average of probabilities
        combined_score = (bert_prob + zero_shot_prob) / 2
        combined_sentiment_label = "positive" if combined_score > 0.5 else "negative"
        return combined_sentiment_label

def analyze_tweet(tweet):
    # Preprocess tweet
    processed_tweet = preprocess_tweet(tweet)

    # Use BERT for sentiment analysis
    inputs = tokenizer(processed_tweet, return_tensors='pt', truncation=True)
    outputs = model(**inputs)
    probs = softmax(outputs.logits, dim=1).detach().numpy()[0]
    sentiment_labels = ['Negative', 'Neutral', 'Positive']
    bert_sentiment = sentiment_labels[probs.argmax()]
    bert_prob = probs.max()

    # Use Zero-Shot Classification
    zero_shot_classification = zero_shot_pipeline(processed_tweet, candidate_labels=["positive", "negative", "neutral"])
    zero_shot_sentiment = zero_shot_classification['labels'][0]
    zero_shot_prob = zero_shot_classification['scores'][0]

    # Determine combined sentiment
    combined_sentiment_label = combined_sentiment(bert_sentiment, zero_shot_sentiment, bert_prob, zero_shot_prob)

    return {
        "tweet": tweet,
        "bert_sentiment": bert_sentiment,
        "zero_shot_classification": zero_shot_sentiment,
        "combined_sentiment": combined_sentiment_label
    }

# Example usage
tweet = "@mini_razdan10 Is this brave true pakistani muzlim issuing fatwa against fake arab muslim. ? Cat calling d kettle ? Stone age mein hi rahega aur bakiyon ko bhi rakhega. Matter of joke for Arabs. Koi nahi sunta in bikhariyon ki. Bhool gaya yeh.. Ki 15 din baad katora lekar wahi jana hai ?"
analysis_result = analyze_tweet(tweet)
print(analysis_result)

