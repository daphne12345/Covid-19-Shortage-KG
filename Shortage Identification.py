# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import warnings
from gensim.models import Word2Vec
from textblob import TextBlob
import re
from nltk.stem import WordNetLemmatizer
import os
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pckl
warnings.filterwarnings(action='ignore')


def __tfidf(in_df, col, covid_paper_terms):
    """
    Calculates TF-IDF on the input dataframe only considering the terms in col.
    :param in_df: input dataframe
    :param col: noun_phrases or entities
    :param covid_paper_terms: covid_paper_terms
    :return: result dataframe and top terms
    """
    vectorizer = TfidfVectorizer(stop_words=covid_paper_terms, max_df=0.9, min_df=5)
    X = vectorizer.fit_transform(
        in_df.groupby('real_month')[col].apply(lambda group: ' '.join(group.apply(' '.join))).to_list())
    feature_array = np.array(vectorizer.get_feature_names())
    df_res = pd.DataFrame(X.toarray())
    df_res.columns = feature_array[df_res.columns]
    df_res['real_month'] = in_df['real_month'].sort_values().unique()
    df_res = df_res.set_index('real_month')
    tw_100 = list(df_res.max().nlargest(100).index)
    return df_res, tw_100


def __closest(given_value, a_list):
    """
    Returns the closest value to the given value in the given list.
    :param given_value:  a value
    :param a_list: a list
    :return: closest value to the input value in the list
    """
    return a_list.index(min(a_list, key=lambda list_value: abs(list_value - given_value)))


def __get_context(text, words, word_context_size=10):
    """
    Returns the noun phrases within the context_window size of a list of words in the text.
    :param text: input text
    :param words: list of words
    :param word_context_size: in words
    :return: list of context noun phrases
    """
    text = text.lower()
    spaces = [0] + [m.start() for m in re.finditer(' ', text)] + [len(text) - 1]
    start_ends = [(__closest(m.start(), spaces) - word_context_size, __closest(m.end(), spaces) + word_context_size) for
                  w in words
                  for m in re.finditer(w, text) if w in text]
    nps = list()
    for (start, end) in start_ends:
        start = 0 if start < 0 else start
        end = len(spaces) - 1 if end >= len(spaces) else end
        nps = nps + list(TextBlob(text[spaces[start]:spaces[end]]).noun_phrases)
    return nps


def get_context_all(in_df, words, word_context_size=30, col='noun_phrases'):
    """
    Retrieves the noun phrases in context of a list of words for the entire input dataframe and weights them by TF-IDF.
    :param in_df: input dataframe
    :param words: list of words to find the context for
    :param word_context_size: the context size in words
    :param col: noun_phrases or entities
    :return: the context dataframe only containing noun phrases retrieved in context and the top 100 terms weighted by TF-IDF
    """
    context = in_df.copy()

    context['noun_phrases'] = context['abstract_processed'].apply(
        lambda text: __get_context(text, words, word_context_size))
    context = context[context['noun_phrases'].apply(len) > 1]

    context, tw_100 = __tfidf(context, col=col)
    return context, tw_100


def __clean_text(text, nps, covid_paper_terms):
    """
    Prepares the text for the embeddding.
    :param text: input text
    :param nps: noun phrases or entities in text (will be connected with _)
    :param covid_paper_terms: covid_paper_terms
    :return: prepared text
    """
    sort_out_regex = re.compile('|'.join(covid_paper_terms))
    text = re.sub(r'[^a-z ]', '', text.lower())
    text = sort_out_regex.sub('', text)
    for np in nps:
        text = text.replace(np, '_'.join(np.lower().replace('the', '').split()))
    wordnet_lemmatizer = WordNetLemmatizer()
    text = [wordnet_lemmatizer.lemmatize(w) for w in text.split(' ')]
    return text


def create_word_embedding(in_df, mode, path):
    """
    Creates a word embedding.
    :param path: path
    :param in_df: input dataframe
    :param mode: noun_phrases or entities
    :return: word embedding
    """
    in_df['text'] = in_df[mode].apply(lambda nps: ' '.join(nps))
    res = in_df.apply(lambda row: __clean_text(row['text'], row[mode]), axis=1)
    model_emb = Word2Vec(sentences=res.to_list(), window=20, min_count=5, workers=4)
    pckl.dump(model_emb, open(path + '/model_emb_' + str(mode) + '.pckl', 'wb'))
    return model_emb


def embedding_most_similar(in_terms, model_emb, covid_paper_terms):
    """
    Retrieves the 100 most similar terms per input terms and returns the 100 most frequent ones.
    :param covid_paper_terms: list of stop words
    :param in_terms: list of input terms
    :param model_emb: word embedding
    :return: 100 top terms
    """
    tw = list()

    terms = list()
    nocovid_terms = [t.replace(' ', '_') for t in in_terms]
    for word in nocovid_terms:
        if word in model_emb.wv:
            terms += pd.DataFrame(model_emb.wv.most_similar(positive=[model_emb.wv[word]], topn=100))[0].to_list()
    word_freq = pd.Series(terms).value_counts().reset_index()
    indices = list(set(word_freq['index']) - set(nocovid_terms + covid_paper_terms))
    word_freq = word_freq[word_freq['index'].isin(indices)].dropna().drop_duplicates()
    tw.append(word_freq.nlargest(100, 0)['index'].to_list())
    return tw


def evaluate(top_terms, shortage_terms_covid):
    """
    Calculate precision, recall and fscore for the list of input terms and the list of shortage terms.
    :param shortage_terms_covid: shortage_terms_covid
    :param top_terms: list of terms to evaluate
    :return: precision, recall, fscore
    """
    retrieved = list(set([w.replace('_', ' ') for w in top_terms]))
    relevant = [w for w in retrieved if any([t in w for t in shortage_terms_covid])]

    precision = len(relevant) / len(retrieved)
    recall = len(relevant) / len(list(set(relevant + shortage_terms_covid)))
    fscore = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    return precision, recall, fscore


# noinspection PyPep8Naming
def start_shortage_identification(df_tm, KG, path, shortage_terms_covid, shortage_terms_nocovid, covid_paper_terms):
    """
    Applies the shortage identification to the dataset once with and once without the KG.
    :param df_tm: dataframe from topic modeling
    :param KG: knowledge graph
    :param path: path
    :param shortage_terms_covid: shortage terms
    :param shortage_terms_nocovid: shortage indicators
    :param covid_paper_terms: list of terms to ignore
    :return: a list of precision, recall and fscore for each of the modes and methods
    """

    if not os.path.exists(path):
        os.makedirs(path)
    os.makedirs(path + '/shortage_identification/')

    rel_terms_in_kg = list(set(KG.s.to_list() + KG.o.to_list()))
    match_str_in_kg = ' '.join(rel_terms_in_kg).replace('_', ' ').replace('  ', ' ')
    entities = [ent.replace('_', ' ').replace('  ', ' ') for ent in rel_terms_in_kg]
    df_tm['entities'] = df_tm['abstract_processed'].apply(lambda text: [ent for ent in entities for ent in text])

    result = list()
    for mode in ['noun_phrases', 'entities']:
        _, top_terms = get_context_all(df_tm, shortage_terms_nocovid, 30, mode)
        result.append([evaluate(top_terms, shortage_terms_covid)])
        model_emb = create_word_embedding(df_tm, mode, path)
        top_terms = embedding_most_similar(shortage_terms_nocovid, model_emb, covid_paper_terms)
        result.append([evaluate(top_terms, shortage_terms_covid)])
    return result
