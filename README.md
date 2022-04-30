# Covid-19-Shortage-KG

1. Download the CORD-19 data from kaggle (only the file "metadata.csv"): https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv
2. Place the metadata file in folder "data".
3. Install requirements:
```
pip -r requirements.txt
```
(if you run into issues installing lda, download  "Microsoft C++ Build Tools": https://visualstudio.microsoft.com/visual-cpp-build-tools/)
4. Download spacy language model:
```
python -m spacy download en_core_web_sm
```
5. To prepare the data run:
```
python data_preparation.py
```
6. To train the hyperparameters run:
```
python tm_hyperparameter_tuning.py
```
7. Check output for hyperparameters, if hyperparameter tuning was not performed, the default parameters can be used.
8. Run topic modeling with the hyperparameters to create the final topic model and reduce the dataset.
```
python topic_modeling.py --alpha 0.03 --beta 0.03 --k 3 --seed_confidence 0.98 --shortage_words = ['goods', 'capacity', 'shortage', 'stock', 'peak', 'deficiency',
                  'market', 'demand', 'inventory', 'reduction', 'resource', 'lack',
                  'manufacturing', 'deficit', 'scarcity', 'product', 'logistics',
                  'unavailability', 'supply chain', 'supply']
``` 
9. To create the Knowledge Graph, run:
```
python kg_creation.py
```
