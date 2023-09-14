from utils import *


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

