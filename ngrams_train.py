from NGramProcessor import *
import os

os.makedirs('language_models', exist_ok=True)

df_train = pd.read_csv('dataset/train_dataset.csv')

df_test = pd.read_csv('dataset/validate_dataset.csv')

unigram = NGramProcessor(df_train, 1)
unigram.train()
unigram.find_probability(save_csv='language_models/unigram.csv')
unigram.calc_perplexity(df_train, save_csv='language_models/unigram_val.csv')
unigram.calc_perplexity(df_train, flag_smooth="turing",
                        save_csv='language_models/unigram_val_turing.csv')

bigram = NGramProcessor(df_train, 2)
bigram.train()
bigram.find_probability(save_csv='language_models/bigram.csv')
bigram.calc_perplexity(df_train, save_csv='language_models/bigram_val.csv')
bigram.calc_perplexity(df_train, flag_smooth="turing",
                       save_csv='language_models/bigram_val_turing.csv')

trigram = NGramProcessor(df_train, 3)
trigram.train()
trigram.find_probability(save_csv='language_models/trigram.csv')
trigram.calc_perplexity(df_train, save_csv='language_models/trigram_val.csv')
trigram.calc_perplexity(df_train, flag_smooth="turing",
                        save_csv='language_models/trigram_val_turing.csv')


quadgram = NGramProcessor(df_train, 4)
quadgram.train()
quadgram.find_probability(save_csv='language_models/quadgram.csv')
quadgram.calc_perplexity(df_train, save_csv='language_models/quadgram_val.csv')
quadgram.calc_perplexity(df_train, flag_smooth="turing",
                         save_csv='language_models/quadgram_val_turing.csv')
