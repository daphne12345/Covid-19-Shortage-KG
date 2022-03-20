# -*- coding: utf-8 -*-
import pandas as pd
import re
from textblob import TextBlob
import spacy
import pickle as pckl
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
from langdetect import detect
import os
os.system('python -m spacy download en_core_web_sm') # restart after running this line
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('brown')
nltk.download('wordnet')


# 1. Read metadata (can be downloaded
# from https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv).
path = ''
df = pd.read_csv(path + 'data/metadata.csv')
df = df[['cord_uid', 'title', 'abstract', 'publish_time']].drop_duplicates()

# 2. convert date to month & year
df = df[df['publish_time'].notna()]  # drop articles without date
df['has_month'] = np.where(df['publish_time'].str.contains('-'), True, False)
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['real_month'] = pd.to_datetime(df['publish_time'].apply(lambda x: x.strftime('%Y-%m')))

# 3. Drop irrelevant data
df = df[(df['real_month'] >= '2019-11') & (
            df['real_month'] <= '2021-10')]  # drop articles before covid & which are not published yet

df = df[df['abstract'].notna()]  # drop articles without an abstract
df = df[df['abstract'].apply(lambda text: len(text.split(' '))) > 50]  # drop articles with a very short abstract

df['abstract_low'] = df['abstract'].apply(
    lambda text: re.sub(r'[^\w]', '', str(text).lower()))  # lowercase and remove special characters
df = df.drop_duplicates(subset='abstract_low')  # drop articles with the same abstract

df = df[df['abstract'].apply(lambda text: detect(text[:50]) == 'en')]  # drop non-english abstracts

# 4. Clean abstracts: lower case and remove special characters
df['abstract_processed'] = df['abstract'].apply(lambda text: re.sub(r'[^\w.!? ]', '', text))

# 5. Extract noun phrases
nlp = spacy.load("en_core_web_sm")
spacy.prefer_gpu()
articles = ['the', 'a', 'an', 'this']
covid_paper_terms = pckl.load(open(path + 'data/covid_paper_terms.pckl', 'rb'))


def extract_noun_phrases(text):
    nps_sents = list()
    nps_all = list()
    for sent in sent_tokenize(text):
        blob = TextBlob(sent).noun_phrases  # extract noun phrases using TextBlob
        blob = [' '.join([n for n in np.lower().split(' ') if n not in articles]) for np in
                blob]  # reomove articles from noun phrases
        spa = [' '.join([n for n in np.text.lower().split(' ') if n not in articles]) for np in nlp(sent).noun_chunks if
               ' '.join([n for n in np.text.split(' ') if
                         n not in articles]) not in blob]  # add spacy noun phrases that are not in the list
        nps = blob + spa

        nps = [re.sub(r'[^\w ]', '', np.strip(' ')) for np in nps]  # sort out special characters
        nps = [np for np in nps if ((np.replace(' ', '_') not in (covid_paper_terms)) & (
                    len(np) > 1))]  # remove general paper terms and covid terms from noun phrases
        nps_sents.append(nps) # per sentence
        nps_all = nps_all + nps # per abstract
    return pd.Series({'np_sent': nps_sents, 'noun_phrases': nps_all})


df[['np_sent', 'noun_phrases']] = df['abstract_processed'].apply(extract_noun_phrases)

# 7. delete irrelevant columns and save preprocessed data
df = df[['cord_uid', 'has_month', 'abstract_processed', 'real_month', 'noun_phrases', 'np_sent']]
df.to_pickle(path + 'data/df_preprocessed.pckl')
