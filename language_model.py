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



df_train = pd.read_csv('dataset/train_dataset.csv')
df_train.dropna(inplace=True)


quadgram = ngrams(df_train,4)
trigram = ngrams(df_train,3)
bigram = ngrams(df_train,2)
unigram = ngrams(df_train,1)



bigram_fd = create_frequency_dict(bigram)
unigram_fd = create_frequency_dict(unigram)
trigram_fd = create_frequency_dict(trigram)
quadgram_fd = create_frequency_dict(quadgram)


find_probability(unigram_fd,unigram_fd,1).to_csv('language_models/unigram.csv',index=False)
find_probability(unigram_fd,bigram_fd,2).to_csv('language_models/bigram.csv',index=False)
find_probability(bigram_fd,trigram_fd,3).to_csv('language_models/trigram.csv',index=False)
find_probability(trigram_fd,quadgram_fd,4).to_csv('language_models/quadgram.csv',index=False)

