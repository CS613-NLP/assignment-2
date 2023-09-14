from NGramProcessor import *


df_train = pd.read_csv('dataset/train_dataset.csv')
# df_train.dropna(inplace=True)

df_test = pd.read_csv('dataset/validate_dataset.csv')
# df_test.dropna(inplace=True)


unigram = NGramProcessor(df_train, df_test, 1)
unigram.find_probability(train=True, save_csv='language_models/unigram.csv')
unigram.find_probability(train=False, save_csv='language_models/unigram_val.csv')


bigram = NGramProcessor(df_train, df_test, 2)
bigram.find_probability(train=True, save_csv='language_models/bigram.csv')
bigram.find_probability(train=False, save_csv='language_models/bigram_val.csv')


trigram = NGramProcessor(df_train, df_test, 3)
trigram.find_probability(train=True, save_csv='language_models/trigram.csv')
trigram.find_probability(train=False, save_csv='language_models/trigram_val.csv')


quadgram = NGramProcessor(df_train, df_test, 4)
quadgram.find_probability(train=True, save_csv='language_models/quadgram.csv')
quadgram.find_probability(train=False, save_csv='language_models/quadgram_val.csv')



