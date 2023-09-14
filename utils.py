import tqdm
import numpy as np
import pandas as pd


def ngrams(df, ngram=1):
    ngrams_corpus = []
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        words = row['Processed_Comment'].split(" ")
        ngrams_sentence = [' '.join(words[i:i+ngram]) for i in range(len(words) - ngram + 1)]
        ngrams_corpus.extend(ngrams_sentence)
    return ngrams_corpus

def create_frequency_dict(ngram):
    frequency_dict = {}
    for item in ngram:
        if item in frequency_dict:
            frequency_dict[item] += 1
        else:
            frequency_dict[item] = 1
    return frequency_dict

def find_probability(frequency_dict_n_1, frequency_dict,n):
    probability_dict={}
    for key in frequency_dict.keys():
        if n==1:
            probability_dict[key]=frequency_dict[key]/sum(frequency_dict.values())
        else:
            key_n_1 = ' '.join(key.split()[:n-1])
            probability_dict[key]=frequency_dict[key]/frequency_dict_n_1[key_n_1]
    df = pd.DataFrame(list(probability_dict.items()), columns=['Phrase', 'Probability'])
    return df