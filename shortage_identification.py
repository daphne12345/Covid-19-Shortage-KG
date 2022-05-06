# -*- coding: utf-8 -*-
import pandas as pd
import warnings
from gensim.models import Word2Vec
import numpy as np
from textblob import TextBlob
import re
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle as pckl


warnings.filterwarnings(action='ignore')

covid_paper_terms = pckl.load(open('data/covid_paper_terms.pckl', 'rb'))


def __tfidf(in_df, col):
    """
    Calculates TF-IDF on the input dataframe only considering the terms in col.
    :param in_df: input dataframe
    :param col: noun_phrases or entities
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


def __clean_text(text, nps):
    """
    Prepares the text for the embedding.
    :param text: input text
    :param nps: noun phrases or entities in text (will be connected with _)
    :return: prepared text
    """
    sort_out_regex = re.compile('|'.join(covid_paper_terms))
    text = re.sub(r"[^a-z ]", '', text.lower())
    text = sort_out_regex.sub('', text)
    for np in nps:
        text = text.replace(np, '_'.join(np.lower().replace('the', '').split()))
    wordnet_lemmatizer = WordNetLemmatizer()
    text = [wordnet_lemmatizer.lemmatize(w) for w in text.split(' ')]
    return text


def create_word_embedding(in_df, mode):
    """
    Creates a word embedding.
    :param in_df: input dataframe
    :param mode: noun_phrases or entities
    :return: word embedding
    """
    in_df['text'] = in_df[mode].apply(lambda nps: ' '.join(nps))
    res = in_df.apply(lambda row: __clean_text(row['text'], row[mode]), axis=1)
    model_emb = Word2Vec(sentences=res.to_list(), window=20, min_count=5, workers=4)
    pckl.dump(model_emb, open('model_emb_' + str(mode) + '.pckl', 'wb'))
    return model_emb


def embedding_most_similar(in_terms, model_emb, covid_paper_terms):
    """
    Retrieves the 100 most similar terms per input terms and returns the 100 most frequent ones.
    :param in_terms: list of input terms
    :param model_emb: word embedding
    :param covid_paper_terms: list of stop words
    :return: 100 top terms
    """
    terms = list()
    nocovid_terms = [t.replace(' ', '_') for t in in_terms]
    for word in nocovid_terms:
        if word in model_emb.wv:
            terms += pd.DataFrame(model_emb.wv.most_similar(positive=[model_emb.wv[word]], topn=100))[0].to_list()
    word_freq = pd.Series(terms).value_counts().reset_index()
    indices = list(set(word_freq['index']) - set(nocovid_terms + covid_paper_terms))
    word_freq = word_freq[word_freq['index'].isin(indices)].dropna().drop_duplicates()
    return word_freq.nlargest(100, 0)['index'].to_list()


def evaluate(top_terms, shortage_terms_covid):
    """
    Calculate precision, recall and fscore for the list of input terms and the list of shortage terms.
    :param top_terms: list of terms to evaluate
    :param shortage_terms_covid: shortage_terms_covid
    :return: precision, recall, fscore
    """
    retrieved = list(set([w.replace('_', ' ') for w in top_terms]))
    relevant = [w for w in retrieved if any([t in w for t in shortage_terms_covid])]

    precision = len(relevant) / len(retrieved) if len(retrieved) > 0 else 0
    recall = len(relevant) / len(list(set(relevant + shortage_terms_covid)))
    fscore = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    print('Precision', precision, 'Recall', recall, 'F-Score', fscore)
    return precision, recall, fscore


def start_shortage_identification(df_tm, KG, shortage_terms_covid, shortage_terms_nocovid):
    """
    Applies the shortage identification to the dataset once with and once without the KG.
    :param df_tm: dataframe from topic modeling
    :param KG: knowledge graph
    :param shortage_terms_covid: shortage terms
    :param shortage_terms_nocovid: shortage indicators
    :return: a list of precision, recall and fscore for each of the modes and methods
    """
    rel_terms_in_kg = list(set(KG.s.to_list() + KG.o.to_list()))
    entities = [ent.replace('_', ' ').replace('  ', ' ') for ent in rel_terms_in_kg]
    pattern = r'\W.*?({})\W.*?'.format('|'.join(entities))
    df_tm['entities'] = df_tm['abstract_processed'].apply(lambda text: re.findall(pattern, text, flags=re.IGNORECASE))

    for mode in ['noun_phrases', 'entities']:
        _, top_terms = get_context_all(df_tm, shortage_terms_nocovid, 30, mode)
        print(mode, 'context')
        evaluate(top_terms, shortage_terms_covid)
        model_emb = create_word_embedding(df_tm, mode)
        top_terms = embedding_most_similar(shortage_terms_nocovid, model_emb, covid_paper_terms)
        print(mode, 'embedding')
        evaluate(top_terms, shortage_terms_covid)


if __name__ == "__main__":
    df_tm = pd.read_pickle('data/df_reduced_by_tm.pckl')
    KG = pd.read_pickle('kgs/KG_final.pckl')
    df_cov = pd.read_csv('data/shortage_terms.csv', delimiter=';')
    shortage_terms_nocovid = df_cov[df_cov['type'].isin(
        ['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])][
        'name'].to_list()
    shortage_terms_covid = df_cov[~df_cov['name'].isin(shortage_terms_nocovid)]['name'].to_list()

    start_shortage_identification(df_tm, KG, shortage_terms_covid, shortage_terms_nocovid)
