from kg_creation import create_kg
from shortage_identification import start_shortage_identification
import pickle as pckl
import os
import pandas as pd
import sys

if __name__ == "__main__":

    path = sys.argv[1] if len(sys.argv) > 1 else ''
    if not os.path.exists(path):
        os.makedirs(path)
    os.makedirs(path + '/kgs')
    os.environ["CORENLP_HOME"] = path + '/corenlp'

    df_tm = pd.read_pickle(path + '/data/df_tm.pckl')
    df_cov = pd.read_csv(path + '/data/shortage_terms.csv', delimiter=';')
    shortage_terms_nocovid = df_cov[df_cov['type'].isin(
        ['product_syn', 'procure_syn', 'shortage_syn', 'stock_syn', 'increase_syn', 'startegy_syn', 'require_syn'])][
        'name'].to_list()
    shortage_terms_covid = df_cov[~df_cov['name'].isin(shortage_terms_nocovid)]['name'].to_list()
    shortage_terms_all = df_cov['name'].to_list()


    KG = create_kg(path, df_tm, shortage_terms_all, df_cov)
    start_shortage_identification(df_tm, KG, path, shortage_terms_covid, shortage_terms_nocovid)
