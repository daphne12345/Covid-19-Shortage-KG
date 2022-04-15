# Context Occurrences weighted by Term Frequency-Inverse Document Frequency
# Word Embedding.

# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings

from gensim.models import Word2Vec

warnings.filterwarnings(action='ignore')
from textblob import TextBlob
import re
from nltk.stem import WordNetLemmatizer

wordnet_lemmatizer = WordNetLemmatizer()
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pckl

os.environ['NUMEXPR_MAX_THREADS'] = '16'
# read data
path = ''
if not os.path.exists(path):
    os.makedirs(path)
os.makedirs(path + '/shortage_identification/')

df_tm = pd.read_pickle(path + '/data/df_tm.pckl')

df_cov = pd.read_csv(path + '/data/shortage_terms.csv', delimiter=';')
shortage_terms_nocovid = df_cov[df_cov['type'].isin(
    ['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])][
    'name'].to_list()
shortage_terms_covid = df_cov[~df_cov['name'].isin(shortage_terms_nocovid)]['name'].to_list()
shortage_terms_all = df_cov['name'].to_list()

covid_paper_terms = pckl.load(open(path + 'data/covid_paper_terms.pckl', 'rb'))


def tfidf(in_df, col):
    vectorizer = TfidfVectorizer(stop_words=cov, max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(
        in_df.groupby('real_month')[col].apply(lambda group: ' '.join(group.apply(' '.join))).to_list())
    feature_array = np.array(vectorizer.get_feature_names())
    df_res = pd.DataFrame(X.toarray())
    df_res.columns = feature_array[df_res.columns]
    df_res['real_month'] = in_df['real_month'].sort_values().unique()
    df_res = df_res.set_index('real_month')
    tw_100 = list(df_res.max().nlargest(100).index)
    return df_res, tw_100


def closest(given_value, a_list):
    return a_list.index(min(a_list, key=lambda list_value: abs(list_value - given_value)))


def get_context(text, words, context_size=10):
    text = text.lower()
    spaces = [0] + [m.start() for m in re.finditer(' ', text)] + [len(text) - 1]
    start_ends = [(closest(m.start(), spaces) - context_size, closest(m.end(), spaces) + context_size) for w in words
                  for m in re.finditer(w, text) if w in text]
    nps = list()
    for (start, end) in start_ends:
        start = 0 if start < 0 else start
        end = len(spaces) - 1 if end >= len(spaces) else end
        nps = nps + list(TextBlob(text[spaces[start]:spaces[end]]).noun_phrases)
    return nps


def get_context_all(in_df, words=shortage_terms_nocovid, word_context_size=3, col='noun_phrases'):
    context = in_df.copy()

    context['noun_phrases'] = context['abstract_processed'].apply(
        lambda text: get_context(text, words, word_context_size))
    context = context[context['noun_phrases'].apply(len) > 1]

    context, tw_100 = tfidf(context, col=col)
    return context, tw_100


def clean_text(text, nps):
    sort_out_regex = re.compile('|'.join(covid_paper_terms))
    text = re.sub(r'[^a-z ]', '', text.lower())
    text = sort_out_regex.sub('', text)
    for np in nps:
        text = text.replace(np, '_'.join(np.lower().replace('the', '').split()))

    text = [wordnet_lemmatizer.lemmatize(w) for w in text.split(' ')]
    return text


def create_word_embedding(in_df, mode):
    in_df['text'] = in_df[mode].apply(lambda nps: ' '.join(nps))
    res = in_df.apply(lambda row: clean_text(row['text'], row[mode]), axis=1)
    model_emb = Word2Vec(sentences=res.to_list(), window=20, min_count=5, workers=4)
    pckl.dump(model_emb, open(path + 'model_emb_' + str(mode) + '.pckl', 'wb'))
    return model_emb


def embedding_most_similar(in_terms, model_emb):
    tw = list()

    terms = list()
    nocovid_terms = [t.replace(' ', '_') for t in in_terms]
    for word in nocovid_terms:
        if word in model_emb.wv:
            terms += pd.DataFrame(model_emb.wv.most_similar(positive=[model_emb.wv[word]], topn=top))[0].to_list()
    word_freq = pd.Series(terms).value_counts().reset_index()
    indices = list(set(word_freq['index']) - set(nocovid_terms + covid_paper_terms))
    word_freq = word_freq[word_freq['index'].isin(indices)].dropna().drop_duplicates()
    tw.append(word_freq.nlargest(100, 0)['index'].to_list())
    return tw


def evaluate(top_terms):
    pass


def start_shortage_identification(mode):
    result = list()
    for mode in ['noun_phrases', 'in_kg']:
        _, top_terms = get_context_all(df_tm, col=mode)
        result.append([evaluate(top_terms)])
        model_emb = create_word_embedding(df_tm, mode)
        top_terms = embedding_most_similar(shortage_terms_nocovid, model_emb)
        result.append([evaluate(top_terms)])
