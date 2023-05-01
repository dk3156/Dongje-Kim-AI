import streamlit as st
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
import pandas as pd
import torch
from finetune import Finetune

#===================================#
"""
Documentation for Milestone 4 - Dongje Kim, dk3156
Incline comments are for the detailed documentation of each part of the code
"""
#===================================#
"""
a list of labels that will be used for the toxicity classification
and dictionary containing scores for the first ten tweets
"""
labels= ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

first_ten_tweets = {
    "Yo bitch Ja Rule is more succe..." : 0.9988,
    "The title is..." : 0.0011,
    "Zawe A..." : 0.0027,
    "If you have a look back at th..." : 0.0005,
    "I don't anonymously edit artic..." : 0.0008,
    "Thank you for understanding..." : 0.0005,
    "Please do not add nonsense to..." : 0.0008,
    "Dear god this site is horribl..." : 0.8693,
    "Only a fool can believe in..." : 0.1242,
    "When..." : 0.0012,
}

"""
Title for the streamlit app and a section for displaying the first ten tweets and the score values
"""
st.title("Sentiment Analysis")

st.title("First Ten Tweets")
for elem in first_ten_tweets:
    st.write("Tweet: ", elem, " with toxicity of", first_ten_tweets[elem])

"""
This function takes in a string text and performs sentiment analysis using a fine-tuned DistilBERT model
for toxicity classification. 
The function first loads the model and tokenizer from the transformers library, 
tokenizes the input text, and then applies the model to generate a prediction. 
The prediction scores for each label are then outputted in the Streamlit app.
"""
def finetune(text):
    model = "dk3156/toxic_tweets_model"
    tokenizer = DistilBertTokenizerFast.from_pretrained(model)
    finetune = DistilBertForSequenceClassification.from_pretrained(model)
    tokens = tokenizer(text, return_tensors="pt")
    res = finetune(**tokens)
    scores = torch.sigmoid(res.logits)
    output = ""
    
    st.write("Tweet: " + text[:20] + "...")
    for i in range(len(labels)):
        st.write(labels[i], " : ", round(scores[0][i].item(), 4))

"""
Gets user inputs and user's selection of three different language model
"""
text = st.text_input("Enter your text", value="Your dress looks like one colorful dishcloth!")
model = st.selectbox("Select the language model", ("Fine-tuned", "Binary","Non-binary"))

"""
loading different models based on user's selection choice and outputting on the Streamlit app.
"""
if st.button('Analyze'):
    if model == "Fine-tuned":
        finetune(text)
    elif model == "Binary":
        classifier = pipeline(model="distilbert-base-uncased-finetuned-sst-2-english", return_all_scores=True)
        result = classifier(text)
        pos = result[0][0]
        neg = result[0][1]
        st.write(pos, neg)
    else:
        classifier = pipeline(model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)
        result = classifier(text)
        for i in range(0,6):   
            st.write(result[0][i])


