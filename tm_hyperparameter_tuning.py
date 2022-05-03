# -*- coding: utf-8 -*-

# This code was used to tune the hyperparameters of the topic model.
import datetime
import numpy as np
import os
import pandas as pd
import pickle as pckl
from hyperactive import Hyperactive
from hyperactive.optimizers import RandomRestartHillClimbingOptimizer
from sklearn.feature_extraction.text import CountVectorizer
from tmtoolkit.topicmod.evaluate import metric_coherence_gensim
import random
import shutil
import lda
import logging

logging.basicConfig(level=logging.WARNING)

# this is a work around to install guided LDA
os.system('git clone https://github.com/dex314/GuidedLDA_WorkAround.git')
lda_path = os.path.abspath(lda.__file__)
lda_path = '/'.join(lda_path.split('/')[:-1]) if '/' in lda_path else '/'.join(lda_path.split('\\')[:-1])

try:
    shutil.move("GuidedLDA_WorkAround/glda_datasets.py", lda_path)
    shutil.move("GuidedLDA_WorkAround/guidedlda.py", lda_path)
    shutil.move("GuidedLDA_WorkAround/guidedutils.py", lda_path)
except:
    pass
from lda import guidedlda as glda

# read data
df = pd.read_pickle('data/df_preprocessed.pckl')

# read shortage lists
df_cov = pd.read_csv('data/shortage_terms.csv', delimiter=';')
shortage_terms_nocovid = df_cov[df_cov['type'].isin(
    ['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])][
    'name'].to_list()
shortage_terms_covid = df_cov[~df_cov['name'].isin(shortage_terms_nocovid)]['name'].to_list()

# Create a sample for the evaluation of the articles.
eval_sample = df.sample(df.shape[0] // 4)
str_nocovid = '|'.join(shortage_terms_nocovid)
str_covid = '|'.join(shortage_terms_covid)

# Create a ground truth with articles conatining a shortage term.
df_ground_truth = eval_sample[eval_sample['abstract_processed'].str.lower().str.contains(str_covid)][['cord_uid']]

vect_eval = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, tokenizer=lambda doc: doc,
                            preprocessor=lambda doc: [x.lower() for x in doc])
X_eval = vect_eval.fit_transform(eval_sample['noun_phrases'])


def optimize(optimizer):
    """
    Function as input for the optimizer. It takes a random sample from the data of 10000 articles and creates 20 topic
    models with different parameters selected by the optimizer on it. For each model the performance is measured and the
    model and its hyperparameters are saved.
    :param optimizer: an instance of the hyperactive optimizer class
    """
    # take a sample and format it for the lda model
    sample = df.sample(10000)
    vectorizer = CountVectorizer(max_df=0.95, min_df=2, max_features=10000, tokenizer=lambda doc: doc,
                                 preprocessor=lambda doc: [x.lower() for x in doc])
    words_all = sample['noun_phrases']
    X = vectorizer.fit_transform(words_all)
    word2id = vectorizer.vocabulary_

    def model(opt):
        """
        Creates and evaluates the model using the input parameters selected by the optimizer.
        :param opt: dictionary of hyperparameters selected by the optimizer
        :return: averaged score of coherence and fscore as optimization value
        """
        # randomly select a given number of seed terms form the shortage indicators
        shortage_words = [x.strip(' ') for x in shortage_terms_nocovid if x.strip(' ') in list(word2id.keys())]
        shortage_words = random.sample(shortage_words, opt['n_seed_words']) if len(shortage_words) > opt[
            'n_seed_words'] else shortage_words

        seed_topic_list = [shortage_words]
        seed_topics = {}
        for t_id, st in enumerate(seed_topic_list):
            for word in st:
                seed_topics[word2id[word]] = t_id

        # create and fit the guided lda model with the input parameters
        lda = glda.GuidedLDA(n_topics=opt['k'], random_state=10, alpha=opt['alpha'], eta=opt['beta'], n_iter=1000)
        lda.fit(X, seed_topics=seed_topics, seed_confidence=opt['seed_confidence'])

        # calculate the average coherence over all topics
        coherences = metric_coherence_gensim(measure='c_v', top_n=1000,
                                             topic_word_distrib=lda.components_, dtm=X,
                                             vocab=np.array([x for x in vectorizer.vocabulary_.keys()]),
                                             texts=words_all.values, return_mean=False)
        coherence = np.mean([c if not np.isnan(c) else 0 for c in coherences])

        # select the top 100 terms of the seeded topic and calulate fscore, precision and recall on them based on the shortage indicators
        keywords = np.array(vectorizer.get_feature_names())
        t_words = keywords.take((-lda.components_[0]).argsort()[:100])

        overlap_nocov = sum([term in ' '.join(t_words) for term in shortage_terms_nocovid])

        precision = overlap_nocov / len(list(set(t_words))) if len(t_words) > 0 else 0
        recall = overlap_nocov / len(shortage_terms_nocovid)
        fscore = 2 * ((precision * recall) / (precision + recall)) if overlap_nocov > 0 else 0

        # save the model and its hyperparamters
        name = datetime.datetime.now().strftime("%m_%d_%H_%M_%S") + '.pckl'
        pckl.dump({'X': X, 'voc': vectorizer.vocabulary_, 'model': lda}, open('models/lda_' + name, 'wb'))
        model_dict = {'name': name, 'coherence': coherence, 'max_overlap': overlap_nocov,
                      'k': opt['k'], 'alpha': opt['alpha'],
                      'beta': opt['beta'], 'seed_confidence': opt['seed_confidence'], 'seed_list': shortage_words,
                      'recall': recall, 'precision': precision, 'fscore': fscore}
        pckl.dump(model_dict, open('models/model_dict_' + name, 'wb'))
        global res
        res = res.append(model_dict, ignore_index=True)

        print(model_dict)
        return (fscore + coherence) / 2

    # search space for the optimizer
    search_space = {
        "k": list(range(3, 7)),
        "alpha": list(np.arange(0.01, 1.0, 0.01)),
        "beta": list(np.arange(0.01, 1.0, 0.01)),
        'seed_confidence': list(np.arange(0.01, 1.0, 0.01)),
        'n_seed_words': list(range(0, len(shortage_terms_nocovid), 1))

    }

    hyper = Hyperactive(verbosity=["progress_bar", "print_results"])
    hyper.add_search(model, search_space, optimizer=optimizer, n_iter=20, max_score=1)
    hyper.run()


if __name__ == "__main__":
    # overall result dataframe to easily compare the models
    res = pd.DataFrame(
        columns=['name', 'coherence', 'max_overlap', 'k', 'alpha', 'beta', 'seed_confidence', 'seed_list',
                 'recall', 'precision', 'fscore'])

    # Run the optimization 5 times, to avoid local optima (total of 100 models)
    for i in range(5):
        optimizer = RandomRestartHillClimbingOptimizer(n_neighbours=8)
        optimize(optimizer)
        res.to_pickle('tm_tuning/tm_tuning_result_' + str(i) + '.pckl')

    # convert the parameters to float
    res['n_seed'] = res['seed_list'].apply(len)
    res['k'] = res['k'].astype(float)
    res['max_overlap'] = res['max_overlap'].astype(float)
    res['precision'] = res['precision'].astype(float)
    res['recall'] = res['recall'].astype(float)
    res['fscore'] = res['fscore'].astype(float)

    # Calculate the average of the fscore and coherence
    res['(Coherence+fscore)/2'] = (res['coherence'] + res['fscore']) / 2

    # create a dataframe of the best performing models
    res_good = res.sort_values(ascending=True, by='(Coherence+fscore)/2').head(10)  # select 10 best models
    res_good = res_good[res_good['seed_confidence'] > 0.9]  # only keep models with a high seed confidence

    k = round(res_good['k'].mean(), 2)
    print('k:', k)

    alpha = round(res_good['alpha'].mean(), 2)
    print('alpha:', alpha)

    beta = round(res_good['beta'].mean(), 2)
    print('beta:', beta)

    n_seed_terms = int(res_good['n_seed_terms'].mean())
    print('Number of seed terms:', n_seed_terms)

    # select the n_seed_terms most occurring seed terms within the best models
    best_terms = res_good['seed_list'].explode().value_counts().head(n_seed_terms).index
    print('Seed terms:', best_terms)

    seed_confidence = int(round(res_good['seed_confidence'].mean(), 2))
    print('Seed confidence:', seed_confidence)
