import streamlit as st
import pandas as pd
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, pipeline
import pandas as pd
import torch
from finetune import Finetune

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

st.title("Sentiment Analysis")

st.title("First Ten Tweets")
for elem in first_ten_tweets:
    st.write("Tweet: ", elem, " with toxicity of", first_ten_tweets[elem])

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

text = st.text_input("Enter your text", value="Your dress looks like one colorful dishcloth!")
model = st.selectbox("Select the language model", ("Fine-tuned", "Binary","Non-binary"))

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


