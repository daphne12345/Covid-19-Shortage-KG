# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import pickle as pckl
from sklearn.feature_extraction.text import CountVectorizer
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
from lda import guidedlda as glda
import logging
import argparse

logging.basicConfig(level=logging.WARNING)

# hyperparameters
parser = argparse.ArgumentParser("simple_example")
parser.add_argument("k", help="Number of topics", type=int, default=3)
parser.add_argument("alpha", help="Alpha", type=int, default=0.03)
parser.add_argument("beta", help="beta", type=int, default=0.03)
parser.add_argument("seed_confidence", help="seed confidence", type=int, default=0.98)
parser.add_argument("seed_terms", help="Seed for the topic to keep", type=int, default=['goods', 'capacity', 'shortage', 'stock', 'peak', 'deficiency',
                  'market', 'demand', 'inventory', 'reduction', 'resource', 'lack',
                  'manufacturing', 'deficit', 'scarcity', 'product', 'logistics',
                  'unavailability', 'supply chain', 'supply'])
args = parser.parse_args()
k = args['k']
alpha = args['alpha']
beta = args['beta']
seed_confidence = args['seed_confidence']
shortage_words = args['seed_terms']

# read data
df = pd.read_pickle('data/df_preprocessed.pckl')

# read shortage lists
df_cov = pd.read_csv('data/shortage_terms.csv', delimiter=';')
shortage_terms_nocovid = df_cov[df_cov['type'].isin(
    ['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])][
    'name'].to_list()
shortage_terms_covid = df_cov[~df_cov['name'].isin(shortage_terms_nocovid)]['name'].to_list()

# Create the vectorizer and word2id dictionary as input for the lda model
vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, tokenizer=lambda doc: doc,
                             preprocessor=lambda doc: [x.lower() for x in doc])
words_all = df['noun_phrases']
X = vectorizer.fit_transform(words_all)
word2id = vectorizer.vocabulary_

# Create seed dictionary seeding topic 0 with the seed terms
shortage_words = [x.strip(' ') for x in shortage_words if x.strip(' ') in list(word2id.keys())]
seed_topic_list = [shortage_words]
seed_topics = {}
for t_id, st in enumerate(seed_topic_list):
    for word in st:
        seed_topics[word2id[word]] = t_id

# create and fit the guided lda model
lda = glda.GuidedLDA(n_topics=k, random_state=10, alpha=alpha, eta=beta, n_iter=1000)
lda.fit(X, seed_topics=seed_topics, seed_confidence=seed_confidence)

###########################
# Evaluation of the model's quality
# calculate the coherence
coherences = metric_coherence_gensim(measure='c_v', top_n=1000,
                                     topic_word_distrib=lda.components_, dtm=X,
                                     vocab=np.array([x for x in vectorizer.vocabulary_.keys()]),
                                     texts=words_all.values, return_mean=False)
coherence = np.mean([c if not np.isnan(c) else 0 for c in coherences])
print('coherence:', coherence)

# Calculate precision, recall and fscore absed on the articles that were selected and the ones that contain a
# shortage term
str_covid = '|'.join(shortage_terms_covid)
df_short = df[df['abstract_processed'].str.lower().str.contains(str_covid)][['cord_uid']]

df['topic'] = np.argmax(lda.transform(X), axis=1)
df_tm = df[df['topic'] == 0]
TP = len(set(df_tm['cord_uid'].to_list()).intersection(set(df_short['cord_uid'].to_list())))
precision = TP / df_tm.shape[0] if df_tm.shape[0] > 0 else 0
recall = TP / df_short.shape[0]
fscore = 2 * ((precision * recall) / (precision + recall)) if TP > 0 else 0

print('precision:', precision)
print('recall:', recall)
print('fscore:', fscore)

# save dataset, model and evaluation measures
df_tm.to_pickle('data/df_reduced_by_tm.pckl')
model_dict = {'model': lda, 'coherence': coherence, 'recall': recall, 'precision': precision, 'fscore': fscore}
pckl.dump(model_dict, open('models/lda_model.pckl', 'wb'))
