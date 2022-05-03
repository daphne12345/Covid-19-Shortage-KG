# -*- coding: utf-8 -*-

import spacy
from bs4 import BeautifulSoup
import pandas as pd
import pickle as pckl
import warnings
import re
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from transformers import LukeTokenizer, LukeForEntityClassification
from sklearn.utils.extmath import softmax
import stanza
from stanza.server import CoreNLPClient
from ampligraph.latent_features import ComplEx, save_model
from ampligraph.evaluation import train_test_split_no_unseen
import requests
from ampligraph.discovery import find_duplicates
import numpy as np
import pandas_dedupe
import swifter

spacy.prefer_gpu()
warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('brown')
nltk.download('omw-1.4')
wordnet_lemmatizer = WordNetLemmatizer()

covid_paper_terms = pckl.load(open('data/covid_paper_terms.pckl', 'rb'))

stanza.install_corenlp()


def clean_kg(KG):
    """
    Syntactically cleans the input knowledge graph.
    :param KG: knowledge graph to clean
    :return: cleaned knowledge graph
    """
    in_shape = KG.shape[0]
    KG['s'] = KG['s'].apply(lambda text: re.sub(r'[^\w ]', '', text)).str.lower()
    KG['p'] = KG['p'].apply(lambda text: re.sub(r'[^\w ]', '', text)).str.lower()
    KG['o'] = KG['o'].apply(lambda text: re.sub(r'[^\w ]', '', text)).str.lower()
    KG = KG[['s', 'p', 'o']].dropna()
    KG = KG[~(KG['s'].str.isnumeric() | KG['o'].str.isnumeric())]
    KG = KG.apply(lambda col: col.str.strip(' ').str.strip('_').str.replace(' ', '_'))
    KG = KG[(KG['s'].apply(len) > 1) & (KG['o'].apply(len) > 1)]
    KG = KG[KG['o'].str.split('_').apply(len) < 7]
    KG = KG[~(KG['s'].isin(covid_paper_terms) | KG['o'].isin(covid_paper_terms))]
    KG['s'] = KG['s'].apply(lambda term: '_'.join(
        [wordnet_lemmatizer.lemmatize(w) for w in nltk.word_tokenize(term.replace('_', ' ')) if
         w not in ['the', 'a', 'an']]))
    KG['p'] = KG['p'].apply(lambda term: wordnet_lemmatizer.lemmatize(term, 'v'))
    KG['o'] = KG['o'].apply(lambda term: '_'.join(
        [wordnet_lemmatizer.lemmatize(w) for w in nltk.word_tokenize(term.replace('_', ' ')) if
         w not in ['the', 'a', 'an']]))
    KG = KG[['s', 'p', 'o']].dropna().drop_duplicates()
    KG = KG[~KG['p'].isin(['owlsameas', 'wikipageusestemplate', 'wikipageexternallink', 'abstract', 'rdfschemacomment',
                           'provwasderivedfrom', 'wikipageid', 'wikipagerevisionid', 'wikipagelength',
                           'wikipageinterlanguagelink', 'thumbnail', 'depiction', 'image', 'genre'])]
    KG = KG[~KG['o'].str.endswith(('svg', 'jpg', 'ppt', 'pdf'))]
    KG = KG[KG['s'] != KG['o']]
    print('Removed:', in_shape - KG.shape[0], 'new shape:', KG.shape[0])
    return KG


def evaluate_kg(KG, rel_terms):
    """
    Evaluates the KG based on the entities that contain a shortage term using prescision and recall.
    :param KG: the knowledge graph
    :param rel_terms: terms that are considered relevant
    :return: precision, recall, number of selected terms, number of selected entities
    """
    rel_terms = [t.replace(' ', '_') for t in rel_terms]
    entities = list(set(KG.s.to_list() + KG.o.to_list()))
    selected_entities = [term for term in entities if any([cov in term for cov in rel_terms])]
    selected_terms = [cov for cov in rel_terms if any([cov in term for term in entities])]
    recall = len(selected_terms) / len(rel_terms)  # how many terms were selected?
    precision = len(selected_entities) / len(entities)  # how many entities are relevant?
    print('precision:', str(precision), 'recall:', str(recall), 'selected terms:', len(selected_terms),
          'selected KG entities:', len(selected_entities))


################################################################################
## Initial Knowledge Graph

def __get_dbpedia_triples_2016(url):
    """
    Retrieves all triples to an entity given by the input URL in the 2016 DBPedia version.
    :param url: URL to the DBPedia entry (current DBPedia)
    :return: all triples to the input entity URL
    """
    url = 'http://dbpedia.mementodepot.org/memento/20161015000000/' + url.replace('/resource/', '/page/')
    r = requests.get(url=url, )
    r.encoding = r.apparent_encoding
    soup = BeautifulSoup(r.text, 'html.parser')
    triples = pd.read_html(str(soup),
                           encoding='utf-8')[0].rename(columns={'Property': 'p', 'Value': 'o'}).dropna()
    triples['s'] = url.split('/')[-1].lower()
    triples['o'] = triples['o'].swifter.apply(
        lambda x: [str(e).split('/')[-1] for e in str(x).split(' ')] if '/' in x else x)
    triples = triples.explode(column='o').dropna().drop_duplicates()
    return triples


def entity_linking(in_df):
    """
    Creates the initial KG from the list of shortage indicators by linking it to DBPedia from 2016 and extracting the
    triples.
    :param in_df: dataframe of shortages and shortage terms
    :return: initial KG
    """
    df_no_cov = in_df[in_df['type'].isin(
        ['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])]
    df_no_cov['link_direct'] = df_no_cov['name'].apply(
        lambda word: 'http://dbpedia.org/resource/' + word.strip(' ').replace(' ', '_').capitalize())
    df_no_cov['link_direct'] = df_no_cov['link_direct'].apply(
        lambda link: link if requests.get(url=link).status_code == 200 else None)
    links = pd.Series(df_no_cov['link_direct'].dropna().explode().dropna().unique())
    KG = pd.concat(links.swifter.apply(__get_dbpedia_triples_2016).to_list())

    KG = clean_kg(KG)
    return KG


def __find_synonyms(word):
    """
    Returns all synonyms to an entity in Wordnet.
    :param word: word to find synonyms for
    :return: Dataframe in the form of KG triples with the relation "same_as" connecting the input word and its synonyms
    """
    word = word.replace('_', ' ')
    synonyms = list(set([s for syns in wn.synsets(word, pos=wn.NOUN) for s in syns.lemma_names() if s != word]))
    if len(synonyms) > 0:
        return pd.DataFrame.from_dict({'s': [word] * len(synonyms), 'p': ['same_as'] * len(synonyms), 'o': synonyms})
    return pd.DataFrame()


def add_synonyms(KG):
    """
    Add synonyms to the KG.
    :param KG: knowledge graph
    :return: KG with synonyms
    """
    KG = KG.append(pd.concat(KG['s'].apply(__find_synonyms).to_list())).drop_duplicates()
    KG = KG.append(pd.concat(KG['o'].apply(__find_synonyms).to_list())).drop_duplicates()
    KG = clean_kg(KG)
    return KG


###################################################################
##Entity Typing

tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")


def __entity_typing(sent, nps):
    """
    Types all noun phrases within a sentence if the probability for a class is higher than 0.3.
    :param sent: a sentence
    :param nps: all noun phrases within that sentence
    :return: list of noun phrases and their type (or None)
    """
    try:
        entity_spans = [[(m.start(), m.end())] for np_ in list(set(nps)) for m in re.finditer(np_, sent) if np_ in sent]
        inputs = tokenizer([sent] * len(entity_spans), entity_spans=entity_spans, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = [
            logit.argmax(-1).item() if softmax(logit.detach().numpy().reshape(1, 9)).max() > 0.3 else None for logit in
            logits]
        pred = [model.config.id2label[e] if e else None for e in predicted_class_idx]
        return list(zip(nps, pred))
    except:
        return None


def __most_occurring(group):
    """
    Returns the majority type per term if the sum of the other types is not higher and if it occurs at least five times.
    :param group: the group of terms with that type
    :return: the majority type or None if the requirements were not met
    """
    vc = group['o'].value_counts()
    if (vc[0] > vc[1:].sum()) & (vc[0] > 5):
        return vc.index[0]
    else:
        return None


def create_entity_type_dict(in_df):
    """
    Creates the look-up table of terms (s) and types (o) with only one type per term.
    :param in_df: reduced dataframe
    :return: the look up table in triple form with the relation entity_type
    """
    in_df['type_sent'] = in_df[['abstract_processed', 'np_sent']].swifter.apply(
        lambda row: [__entity_typing(sent, nps) for (sent, nps) in
                     zip(sent_tokenize(row['abstract_processed']), row['np_sent'])], axis=1)
    entity_types = pd.DataFrame(in_df['type_sent'].explode().explode().to_list()).rename(columns={0: 's', 1: 'o'})
    entity_types = entity_types[entity_types['o'].notna()]
    entity_types = entity_types[entity_types['s'].apply(len) > 2]
    entity_types = entity_types.groupby('s').swifter.apply(__most_occurring).dropna().reset_index()

    entity_types['p'] = 'entity_type'
    entity_types = clean_kg(entity_types)
    entity_types[['s', 'p', 'o']].to_pickle('data/entity_types.pckl')
    return entity_types


def add_entity_types(KG, entity_types=None):
    """
    Adds the types to the KG from the look-up table where the entity is part of the KG.
    :param KG: input knowledge graph
    :param entity_types: the look-up table of entity types (is read from file if None)
    :return: KG enhanced by entity types
    """
    if not entity_types:
        entity_types = pd.read_pickle('data/entity_types.pckl')
    entity_types = entity_types[entity_types['s'].isin(KG['s'].unique()) | entity_types['s'].isin(KG['o'].unique())]
    KG = KG.append(entity_types)[['s', 'p', 'o']]
    KG = clean_kg(KG)
    return KG


######################################################
## Relation Extraction


def __extract_verbs(sent, nps):
    """
    Extracts the verbs in between two noun phrases as relation.
    :param sent: sentence
    :param nps: list of noun phrases in that sentence
    :return: list of noun phrases connected by a verb found within the sentence in KG format (s,p,o)
    """
    try:
        start_end = sorted(list(
            set([(m.start(), m.end()) for np in nps for m in re.finditer(np, sent) if ((np in sent) & (len(np) > 2))])),
            key=lambda k: k[0])
        rel_list = list()
        for i in range(len(start_end)):
            for (start2, end2) in start_end[i + 1:]:
                if (start2 - start_end[i][0]) < 50:
                    verbs = [word for word in word_tokenize(sent[start_end[i][0]:start2]) if
                             pos_tag([word])[0][1].startswith('V')]
                    for verb in verbs:
                        rel_list.append({'s': sent[start_end[i][0]:start_end[i][1]], 'p': verb, 'o': sent[start2:end2],
                                         'sent': sent})
        return rel_list
    except:
        return list()


def __correct_annotation(abstract, annotation, nps):
    """
    Corrects the annotation made by core nlp. If no triples were extracted for a sentence, the extract_verb function is called instead.
    :param abstract: input abstract that was annotated
    :param annotation: the annotations made for the abstract
    :param nps: the noun phrases in that abstract
    :return: list of subject, relation, object and originating sentence
    """
    relations_list = []
    for text, sent, np in zip(sent_tokenize(abstract), annotation.sentence, nps):
        np = [n.lower() for n in np]
        triples = sent.openieTriple
        if len(triples) == 0:
            [relations_list.append(triple) for triple in __extract_verbs(text, np)]
        else:
            for triple in triples:
                # noun phrases or preposition so that no verbs are in the subject & objects
                sub = ' '.join([word for word in word_tokenize(triple.subject) if (
                        pos_tag([word.lower()])[0][1].startswith('N') or pos_tag([word.lower()])[0][1].startswith(
                    'IN'))])
                obj = ' '.join([word for word in word_tokenize(triple.object) if (
                        pos_tag([word.lower()])[0][1].startswith('N') or pos_tag([word.lower()])[0][1].startswith(
                    'IN'))])

                if (len(sub) > 0) & (len(obj) > 0):
                    relations_list.append({'s': sub, 'p': triple.relation, 'o': obj, 'sent': text})
    return pd.DataFrame(relations_list)


def __create_triples(in_df, max_entailments=600):
    """
    Uses Core NLP to annotate the abstracts in the input dataframe with extracted triples.
    :param in_df: input dataframe to annotate
    :param max_entailments: Maximum entailments per clause
    :return: Annotated dataframe
    """
    rel_df = pd.DataFrame()
    sam_list = np.array_split(in_df, 20)
    for small_sample in sam_list:
        with CoreNLPClient(annotators=['openie'], memory='16G', endpoint='http://localhost:9002', be_quiet=True,
                           use_gpu=True, properties={'openie.max_entailments_per_clause': max_entailments}) as client:
            small_sample['annotated'] = small_sample['abstract_processed'].swifter.apply(client.annotate)
            rel_df = rel_df.append(pd.concat(small_sample.swifter.apply(
                lambda row: __correct_annotation(row['abstract_processed'], row['annotated'], row['np_sent']),
                axis=1).to_list()))
    return rel_df


def __longest_head_tail(KG):
    """
    Returns only the longest head (tail) for triples with the same relation and tail (head).
    :param KG: knowledge graph
    :return: reduced KG
    """
    KG['sent'] = KG['sent'].astype(str)
    rel_df_grouped = KG.groupby(['s', 'p', 'sent'])['o'].apply(
        lambda group: max([(val, len(val)) for val in group], key=lambda x: x[1])[0]).reset_index()
    KG = rel_df_grouped.merge(KG[['s', 'p', 'sent']], on=['s', 'p', 'sent']).drop_duplicates()[['s', 'p', 'o', 'sent']]
    rel_df_grouped = KG.groupby(['p', 'o', 'sent'])['s'].apply(
        lambda group: max([(val, len(val)) for val in group], key=lambda x: x[1])[0]).reset_index()
    return rel_df_grouped.merge(KG[['p', 'o', 'sent']], on=['p', 'o', 'sent']).drop_duplicates()[
        ['s', 'p', 'o', 'sent']]


def __split_to_sentences(in_kg, in_df):
    """
    Splits the dataframe into sentences and only keeps entries that contain a KG entity.
    :param in_kg: string of entities in the KG separated by a |
    :param in_df: reduced dataframe
    :return: the dataframe in sentence format containing only sentences that contain a KG entity
    """
    df_tm_sent = in_df[in_df['abstract_processed'].str.lower().str.contains(in_kg)]
    df_tm_sent['sent'] = df_tm_sent['abstract_processed'].swifter.apply(sent_tokenize)
    df_tm_sent['both'] = df_tm_sent.apply(lambda row: list(zip(row['sent'], row['np_sent'])), axis=1)
    df_tm_sent = df_tm_sent.drop(columns=['abstract_processed', 'noun_phrases', 'np_sent', 'sent'])
    df_tm_sent = df_tm_sent.explode(column='both')
    df_tm_sent['sent'] = df_tm_sent['both'].apply(lambda x: x[0])
    df_tm_sent['np_sent'] = df_tm_sent['both'].apply(lambda x: x[1])
    df_tm_sent = df_tm_sent[df_tm_sent['sent'].str.lower().str.contains(in_kg)].drop(columns='both')
    df_tm_sent = df_tm_sent.groupby(['cord_uid']).agg({'sent': lambda x: '.'.join(x), 'np_sent': lambda x: x.tolist()},
                                                      axis=1).reset_index()
    df_tm_sent.rename(columns={'sent': 'abstract_processed'}, inplace=True)
    return df_tm_sent


def relation_extraction(KG, in_df):
    """
    Applies relation extraction to the data and adds only triples where at least one entity is already part of the KG.
    :param in_df: reduced dataframe
    :param KG: knowledge graph
    :return: KG with triples from text
    """
    in_kg = '|'.join(set(KG['s'].to_list() + KG['o'].to_list())).replace('_', ' ')
    df_tm_sent = __split_to_sentences(in_kg, in_df)
    rel_df_sent = __create_triples(df_tm_sent)
    rel_df_sent = __longest_head_tail(rel_df_sent)

    # remove subject and object with same relation
    rel_df_sent = rel_df_sent.groupby(['s', 'o', 'sent']).apply(
        lambda group: min(list(zip(group['p'], group['p'].apply(len))), key=lambda k: k[1])[0]).reset_index().rename(
        columns={0: 'p'})

    # only keep triples where at least one is already in KG
    triples_to_add = rel_df_sent[rel_df_sent['s'].str.contains(in_kg) | rel_df_sent['o'].str.contains(in_kg)][
        ['s', 'p', 'o']]

    KG = KG.append(triples_to_add).drop_duplicates()
    KG = clean_kg(KG)
    return KG


########################################
# superclasses


nlp = spacy.load("en_core_web_sm")


def __get_root(entity):
    """
    Returns the roots to all noun phrases within the input entity.
    :param entity: input entity
    :return: list of roots
    """
    res = list()
    doc = nlp(entity.replace('_', ' '))
    for np in doc.noun_chunks:
        res.append(str(np.root))
    return res


def create_superclasses(KG):
    """
    Creates superclasses by extracting the root of noun phrases and adding them to the KG if more than 10 entities 
    have that root. The relation is subclass_of.
    :param KG: knowledge graph
    :return: knowledge graph with superclasses
    """
    s_o = KG['s'].append(KG['o']).drop_duplicates().reset_index()
    s_o['root'] = s_o[0].swifter.apply(__get_root)
    s_o = s_o.explode(column='root')
    s_o = s_o[s_o[0] != 'root']

    # check if at least 10 entities have that root and that the roots are longer than 2
    cts = s_o['root'].value_counts() >= 10
    terms = pd.Series(cts[cts].index)
    terms = terms[terms.apply(len) > 2]

    # add roots to the KG 
    s_o = s_o[s_o['root'].isin(terms)]
    to_add_s = pd.DataFrame({'s': s_o[0], 'p': ['subclass_of'] * s_o.shape[0], 'o': s_o['root']})
    to_add_s = clean_kg(to_add_s)
    KG = KG.append(to_add_s)
    KG = clean_kg(KG)
    return KG


##########################################
# removing semantic duplicates (not used, since relevant triples were removed)

def __create_embedding(KG):
    """
    Creates an embedding of the KG with always the same relation using Complex.
    :param KG: knowledge graph
    :return: the KG as a numpy array with the relation related_to for all triples and the KG embedding
    """
    KG['p'] = 'related_to'
    KG = clean_kg(KG)
    X = KG[['s', 'p', 'o']].to_numpy()
    X_train, X_test = train_test_split_no_unseen(X, test_size=int(X.shape[0] * 0.3), allow_duplication=True)
    model = ComplEx()
    model.fit(X_train)
    save_model(model, 'KGs/kg_embedding.pckl')
    return X, model


def remove_duplicates(KG):
    """
    Removes duplicate triples in the KG based on the embedding.
    :param KG: knowledge graph
    :return: KG without duplicates
    """
    X, model = __create_embedding(KG)
    dups, _ = find_duplicates(X, model, mode='triple', tolerance=0.3)
    dups = pd.DataFrame(dups)
    dups = dups[dups[1].notna()]
    to_remove = dups.apply(list, axis=1).explode().apply(pd.Series).dropna().drop_duplicates().rename(
        columns={0: 's', 1: 'p', 2: 'o'})

    KG = pd.concat([KG, to_remove]).drop_duplicates(keep=False)
    KG = clean_kg(KG)
    return KG


# !pip install numpy --upgrade --ignore-installed
def remove_duplicates_dedupe(KG):
    """
    REQUIRES USER INPUT
    Removes duplicates from the KG using Dedupe.
    :param KG: knowledge graph
    :return: KG without duplciates
    """
    KG = KG[['s', 'p', 'o']].drop_duplicates()
    variables = [['s', 'String'], ['p', 'String'], ['o', 'String']]
    kg_deduped = pandas_dedupe.dedupe_dataframe(KG, variables)

    keep = kg_deduped[kg_deduped['confidence'] == 1]
    kg_deduped = kg_deduped[~kg_deduped.index.isin(keep.index)]
    unique = kg_deduped['cluster id'].value_counts() == 1
    unique = unique[unique].index
    keep = keep.append(kg_deduped[kg_deduped['cluster id'].isin(unique)])

    # keep longer triple per cluster and append to keepers
    duplicates = kg_deduped[~kg_deduped.index.isin(keep.index)]
    duplicates['all'] = duplicates['s'] + duplicates['p'] + duplicates['o']
    duplicates = duplicates[duplicates['all'].notna()]
    longer_dup = duplicates.groupby('cluster id').apply(lambda group: group['all'].apply(len).idxmax()).values
    duplicates = duplicates[duplicates.index.isin(longer_dup)]
    KG = keep[['s', 'p', 'o']].append(duplicates)[['s', 'p', 'o']]
    KG = clean_kg(KG.dropna())
    return KG


def create_kg(df_tm, df_cov):
    """
     Creates the KG.
    :param df_tm: reduced dataframe from topic modeling
    :param df_cov: dataframe of shortage terms and shortage indicators
    :return: KG
    """
    shortage_terms_all = df_cov['name'].to_list()
    KG = entity_linking(df_cov)
    print('Entity Linking')
    evaluate_kg(KG, shortage_terms_all)

    KG = add_synonyms(KG)
    print('Synonyms')
    evaluate_kg(KG, shortage_terms_all)

    entity_types = create_entity_type_dict(df_tm)
    KG = add_entity_types(KG, entity_types)
    print('Entity Types')
    evaluate_kg(KG, shortage_terms_all)

    KG = relation_extraction(KG, df_tm)
    print('Relation Extraction')
    evaluate_kg(KG, shortage_terms_all)

    KG = relation_extraction(KG, df_tm)
    print('2nd Relation Extraction')
    evaluate_kg(KG, shortage_terms_all)

    KG = create_superclasses(KG)
    print('Superclasses')
    evaluate_kg(KG, shortage_terms_all)

    KG = add_entity_types(KG, entity_types)
    print('2nd Entity Types (Final KG)')
    evaluate_kg(KG, shortage_terms_all)

    KG.to_pickle('kgs/KG_final.pckl')


#######################################
# Create Knowledge Graph
if __name__ == "__main__":
    df_tm = pd.read_pickle('data/df_reduced_by_tm.pckl')
    df_cov = pd.read_csv('data/shortage_terms.csv', delimiter=';')

    create_kg(df_tm, df_cov)
