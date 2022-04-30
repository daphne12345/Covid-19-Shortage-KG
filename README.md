# Covid-19-Shortage-KG

1. Download the CORD-19 data from kaggle (only the file "metadata.csv"): https://www.kaggle.com/datasets/allen-institute-for-ai/CORD-19-research-challenge?select=metadata.csv
2. Place file in folder "data".
3. Install requirements:
```
pip -r requirements.txt
```
4. Download spacy model
```
python -m spacy download en_core_web_sm
```
5. To prepare the data run:
```
python Data_Preparation.py
```
