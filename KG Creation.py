# -*- coding: utf-8 -*-

import spacy

spacy.prefer_gpu()
import nltk
from nltk.corpus import wordnet as wn
# !pip install SPARQLWrapper
from SPARQLWrapper import SPARQLWrapper, JSON
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle as pckl
import datetime
import os
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import csv
import re
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('brown')
from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer = WordNetLemmatizer()
from textblob import TextBlob
from nltk.tokenize import word_tokenize, sent_tokenize, pos_tag
from transformers import LukeTokenizer, LukeForEntityClassification
from sklearn.utils.extmath import softmax
import stanza
from stanza.server import CoreNLPClient

# read data
path = ''
if not os.path.exists(path):
    os.makedirs(path)
    os.makedirs(path + 'kgs')

df_tm = pd.read_pickle(path + '/data/df_tm.pckl')

# read shortage lists
df_cov = pd.read_csv(path + '/data/shortage_terms.csv', delimiter=';')
shortage_terms_nocovid = df_cov[df_cov['type'].isin(
    ['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])][
    'name'].to_list()
shortage_terms_covid = df_cov[~df_cov['name'].isin(shortage_terms_nocovid)]['name'].to_list()
shortage_terms_all = df_cov['name'].to_list()


#clean KG
covid_paper_terms = pckl.load(open(path + 'data/covid_paper_terms.pckl', 'rb'))

def clean_kg(kg):
  in_shape = kg.shape[0]
  if 'real_month' not in kg.columns:
    kg['real_month'] = datetime.datetime(2000,1,1)
  kg['real_month'] = kg['real_month'].fillna(datetime.datetime(2000,1,1))
  kg['s'] = kg['s'].apply(lambda text: re.sub(r'[^\w ]', '', text)).str.lower()
  kg['p'] = kg['p'].apply(lambda text: re.sub(r'[^\w ]', '', text)).str.lower()
  kg['o'] = kg['o'].apply(lambda text: re.sub(r'[^\w ]', '', text)).str.lower()
  kg = kg[['s','p','o', 'real_month']].dropna()
  kg = kg[~(kg['s'].str.isnumeric() | kg['o'].str.isnumeric())]
  kg = kg.apply(lambda col: col.str.strip(' ').str.strip('_').str.replace(' ','_') if col.name!='real_month' else col)
  kg = kg[(kg['s'].apply(len)>1)&(kg['o'].apply(len)>1)]
  kg = kg[kg['o'].str.split('_').apply(len)<7]
  kg = kg[~(kg['s'].isin(covid_paper_terms) | kg['o'].isin(covid_paper_terms))]
  kg['s'] = kg['s'].apply(lambda term: '_'.join([wordnet_lemmatizer.lemmatize(w) for w in nltk.word_tokenize(term.replace('_',' ')) if w not in ['the','a','an']]))
  kg['p'] = kg['p'].apply(lambda term: wordnet_lemmatizer.lemmatize(term,'v'))
  kg['o'] = kg['o'].apply(lambda term: '_'.join([wordnet_lemmatizer.lemmatize(w) for w in nltk.word_tokenize(term.replace('_',' ')) if w not in ['the','a','an']]))
  kg = kg[['s','p','o', 'real_month']].dropna().drop_duplicates(subset=['s','p','o'])
  kg = kg[~kg['p'].isin(['owlsameas', 'wikipageusestemplate',  'wikipageexternallink','abstract','rdfschemacomment', 'provwasderivedfrom', 'wikipageid', 'wikipagerevisionid', 'wikipagelength','wikipageinterlanguagelink','thumbnail', 'depiction', 'image', 'genre'])]
  kg = kg[~kg['o'].str.endswith(('svg','jpg','ppt','pdf'))]
  kg = kg[kg['s']!=kg['o']]
  print('Dropped:', in_shape-kg.shape[0], 'new shape:', kg.shape[0])
  return kg

################################################################################
## Initial Knowledge Graph

#get all triples for a linked entity
def get_dbpedia_triples(uri):
  sparql = SPARQLWrapper("http://dbpedia.org/sparql")
  sparql.setQuery("""DESCRIBE  <"""+ str(uri)+""">""")
  sparql.setReturnFormat(JSON)
  results = sparql.query().convert()
  df_res = pd.DataFrame(results['results']['bindings']).dropna()
  df_o = df_res['o'].apply(pd.Series)
  if 'lang' in df_o.columns:
    df_res['lang'] = df_o['lang']
    df_res = df_res[df_res['lang'].isin([np.nan, 'en'])]
  else:
    df_res['lang'] = None
  df_res['p'] = df_res['p'].apply(pd.Series)['value'].apply(lambda val: str(val).split('/')[-1])
  df_res['o'] = df_o['value'].apply(lambda val: str(val).split('/')[-1])
  df_res['s'] = df_res['s'].apply(pd.Series)['value'].apply(lambda val: str(val).split('/')[-1])
  #df_res = df_res[df_res['p'].isin(keep)].drop_duplicates().dropna()
  return df_res


def get_dbpedia_triples_2016(url):
  url = 'http://dbpedia.mementodepot.org/memento/20161015000000/' + url.replace('/resource/', '/page/')
  r = requests.get(url = url,)
  r.encoding = r.apparent_encoding
  soup = BeautifulSoup(r.text, 'html.parser')
  triples = pd.read_html(str(soup),
  encoding='utf-8')[0].rename(columns={'Property':'p','Value':'o'}).dropna()
  triples['s'] = url.split('/')[-1].lower()
  triples['real_month'] = datetime.datetime(2016,10,15)
  triples['o'] = triples['o'].apply(lambda x: [str(e).split('/')[-1] for e in str(x).split(' ')] if '/' in x else x)
  triples = triples.explode(column='o').dropna().drop_duplicates()
  return triples


df_no_cov = df_cov[df_cov['type'].isin(['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])]

#link general shortage terms per direct matching to dbpedia (more matches and also exact matches)
df_no_cov = df_cov[df_cov['type'].isin(['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])]
df_no_cov['link_direct'] = df_no_cov['name'].apply(lambda word: 'http://dbpedia.org/resource/' + word.strip(' ').replace(' ','_').capitalize())
df_no_cov['link_direct'] = df_no_cov['link_direct'].apply(lambda link: link if requests.get(url = link).status_code==200 else None)
links = pd.Series(df_no_cov['link_direct'].dropna().explode().dropna().unique())


#get all triples in dbpedia for each linked entity (2016)
KG = pd.concat(links.apply(get_dbpedia_triples_2016).to_list())
KG = clean_kg(KG)


### add synonyms from wordnet for general covid terms
def add_synonyms(word):
  word = word.replace('_',' ')
  synonyms = list(set([s for syns in wn.synsets(word, pos=wn.NOUN) for s in syns.lemma_names() if s!=word]))
  if len(synonyms)>0:
    return pd.DataFrame.from_dict({'s':[word]*len(synonyms), 'p':['same_as']*len(synonyms), 'o':synonyms})
  return pd.DataFrame()


KG = KG.append(pd.concat(df_no_cov['name'].apply(add_synonyms).to_list())).drop_duplicates()
KG = clean_kg(KG)

KG.to_pickle(path + 'kgs/initial_kg.pckl')

###################################################################
##Entity Typing

tokenizer = LukeTokenizer.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")
model = LukeForEntityClassification.from_pretrained("studio-ousia/luke-large-finetuned-open-entity")


def ner(sent, nps):
    try:
        entity_spans = [[(m.start(),m.end())] for np_ in list(set(nps)) for m in re.finditer(np_, sent) if np_ in sent]
        inputs = tokenizer([sent]*len(entity_spans), entity_spans=entity_spans, return_tensors="pt")
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = [logit.argmax(-1).item() if softmax(logit.detach().numpy().reshape(1,9)).max()>0.3 else None for logit in logits]
        pred = [model.config.id2label[e] if e else None  for e in predicted_class_idx]
        return list(zip(nps,pred))
    except:
        return None

df_tm['type_sent'] = df_tm[['abstract_processed','np_sent']].apply(lambda row: [ner(sent,nps) for (sent,nps) in zip(sent_tokenize(row['abstract_processed']),row['np_sent'])], axis=1)
df_tm.to_pickle(path + '/data/ner_df_tm.pckl')

luke_types = pd.DataFrame(df_tm['type_sent'].explode().explode().to_list()).rename(columns={0:'term', 1:'type'})
luke_types = luke_types[luke_types['type'].notna()]
luke_types = luke_types[luke_types['term'].apply(len)>2]


def count_occ(group):
  vc = group['type'].value_counts()
  if (vc[0]>vc[1:].sum())&(vc[0]>5):
    return vc.index[0]
  else:
    return None

#get most frequent types for each term in df_tm
luke_types = luke_types.groupby('term').apply(count_occ).dropna().reset_index()
luke_types['term'] = luke_types['term'].str.lower().str.strip(' ').str.replace(' ','_')
luke_types = luke_types[~luke_types['term'].str.isnumeric()].rename(columns={0:'type'})
luke_types.to_csv(path + 'kgs/luke_kg_02_01_2022.csv', index=False)

#add type as triple to KG
def add_luke_types(in_kg):
  luke_types = pd.read_csv(path + 'kgs/luke_kg_02_01_2022.csv').rename(columns={'0':'type'})
  luke_types = luke_types[luke_types['term'].isin(in_kg['s'].unique())|luke_types['term'].isin(in_kg['o'].unique())]
  luke_types['p'] = 'luke_type'
  luke_types = luke_types.rename(columns={'term':'s','type':'o'})
  in_kg = in_kg.append(luke_types)[['s','p','o']]
  return in_kg

KG = add_luke_types(KG)
KG = clean_kg(KG)
KG.to_pickle(path + 'kgs/kg_ner.pckl')


######################################################
## Relation Extraction

os.environ["CORENLP_HOME"] = path + '/corenlp'
def extract_verbs(sent, nps):
  # start_end = sorted(list(set([(m.start(),m.end()) for np in nps for m in re.finditer(np, sent) if ((np in sent)&(len(np)>2))])), key= lambda k: k[0])
  try:
      start_end = sorted(list(set([(m.start(),m.end()) for np in nps for m in re.finditer(np, sent) if ((np in sent)&(len(np)>2))])), key= lambda k: k[0])
      rel_list = list()
      for i in range(len(start_end)):
        for (start2,end2) in start_end[i+1:]:
          if (start2-start_end[i][0])<50:
            verbs = [word for word in word_tokenize(sent[start_end[i][0]:start2]) if pos_tag([word])[0][1].startswith('V') ]
            for verb in verbs:
              rel_list.append({'s': sent[start_end[i][0]:start_end[i][1]], 'p': verb, 'o': sent[start2:end2], 'sent':sent})
      return rel_list
  except:
      print(nps)
      print(sent)
      return list()


def correct_annotation(abstract, annotation, nps):
    relations_list = []
    for text, sent,np in zip(sent_tokenize(abstract), annotation.sentence,nps):
        np = [n.lower() for n in np]
        triples = sent.openieTriple
        if len(triples)==0:
            [relations_list.append(triple) for triple in extract_verbs(text, np)]
        else:
            for triple in triples:
                #noun phrases or preposition so that no verbs are in the subject & objects
                sub = ' '.join([word for word in word_tokenize(triple.subject) if (pos_tag([word.lower()])[0][1].startswith('N') or pos_tag([word.lower()])[0][1].startswith('IN'))])
                obj = ' '.join([word for word in word_tokenize(triple.object) if (pos_tag([word.lower()])[0][1].startswith('N') or pos_tag([word.lower()])[0][1].startswith('IN'))])

                if (len(sub)>0) &(len(obj)>0):
                    relations_list.append({'s': sub , 'p': triple.relation, 'o': obj, 'sent':text})
    return pd.DataFrame(relations_list)


def create_triples(sample, num=600):
    rel_df = pd.DataFrame()
    sam_list = np.array_split(sample, 20)
    for small_sample in sam_list:
        with CoreNLPClient(annotators=['openie'], memory='16G', endpoint='http://localhost:9002', be_quiet=True, use_gpu=True, properties={'openie.max_entailments_per_clause':num}) as client:
            small_sample['annotated'] = small_sample['abstract_processed'].apply(client.annotate)
            rel_df = rel_df.append(pd.concat(small_sample.apply(lambda row: correct_annotation(row['abstract_processed'], row['annotated'], row['np_sent']),axis=1).to_list()))
    return rel_df


def longest_head_tail(kg):
    kg['sent'] = kg['sent'].astype(str)
    rel_df_grouped = kg.groupby(['s','p', 'sent'])['o'].apply(lambda group: max([(val, len(val)) for val in group], key=lambda x: x[1])[0]).reset_index()
    kg = rel_df_grouped.merge(kg[['s', 'p','sent']], on=['s','p', 'sent']).drop_duplicates()[['s','p','o','sent']]
    rel_df_grouped = kg.groupby(['p', 'o', 'sent'])['s'].apply(lambda group: max([(val, len(val)) for val in group], key=lambda x: x[1])[0]).reset_index()
    return rel_df_grouped.merge(kg[['p', 'o', 'sent']], on=['p','o', 'sent']).drop_duplicates()[['s', 'p', 'o', 'sent']]


in_kg = '|'.join(set(KG['s'].to_list() + KG['o'].to_list())).replace('_',' ')

# #sent contains shortage
df_tm_sent = df_tm[df_tm['abstract_processed'].str.lower().str.contains(in_kg)]
df_tm_sent['sent'] = df_tm_sent['abstract_processed'].apply(sent_tokenize)
df_tm_sent['both'] = df_tm_sent.apply(lambda row: list(zip(row['sent'],row['np_sent'])), axis=1)
df_tm_sent = df_tm_sent.drop(columns=['abstract_processed','noun_phrases', 'np_sent', 'sent'])
df_tm_sent = df_tm_sent.explode(column='both')
df_tm_sent['sent'] = df_tm_sent['both'].apply(lambda x: x[0])
df_tm_sent['np_sent'] = df_tm_sent['both'].apply(lambda x: x[1])
df_tm_sent = df_tm_sent[df_tm_sent['sent'].str.lower().str.contains(in_kg)].drop(columns='both')
df_tm_sent = df_tm_sent.groupby(['cord_uid']).agg({'sent': lambda x: '.'.join(x),'np_sent':lambda x: x.tolist()},axis=1).reset_index()
df_tm_sent.rename(columns={'sent':'abstract_processed'}, inplace=True)

rel_df_sent = create_triples(df_tm_sent)
rel_df_sent.to_pickle(path + 'output/df_tm_triples_sent_strict_wn_no_covid_2016_final.pckl')

rel_df_sent = longest_head_tail(rel_df_sent)
#remove subject and object with same relation
rel_df_sent = rel_df_sent.groupby(['s','o', 'sent']).apply(lambda group: min(list(zip(group['p'],group['p'].apply(len))),key=lambda k: k[1])[0]).reset_index().rename(columns={0:'p'})

#only keep triples where at least one is already in KG
triples_to_add = rel_df_sent[rel_df_sent['s'].str.contains(in_kg)|rel_df_sent['o'].str.contains(in_kg)][['s','p','o']]
KG = KG.append(triples_to_add).drop_duplicates()
KG.to_pickle(path + 'kgs/kg_re.pckl')

##################################################

