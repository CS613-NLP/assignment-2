from NGramProcessor import *
import os

os.makedirs('language_models', exist_ok=True)
# os.makedirs('language_models/laplace', exist_ok=True)

df_train = pd.read_csv('dataset/train_dataset.csv')

df_test = pd.read_csv('dataset/validate_dataset.csv')

unigram = NGramProcessor(df_train, 1)
unigram.train()
unigram.find_probability(save_csv='language_models/unigram.csv')
unigram.calc_perplexity(df_train, save_csv='language_models/unigram_perplexity.csv')
unigram.calc_perplexity(df_train, smoothing="laplace",
                        save_csv='language_models/laplace/unigram_perplexity_laplace_train.csv', log_prob_save_csv='language_models/laplace/unigram_log_prob_laplace_train.csv')
unigram.calc_perplexity(df_test, smoothing="laplace",
                        save_csv='language_models/laplace/unigram_perplexity_laplace_test.csv', log_prob_save_csv='language_models/laplace/unigram_log_prob_laplace_test.csv')


bigram = NGramProcessor(df_train, 2)
bigram.train()
bigram.find_probability(save_csv='language_models/bigram.csv')
bigram.calc_perplexity(df_train, save_csv='language_models/bigram_perplexity.csv')
bigram.calc_perplexity(df_train, smoothing="laplace",
                        save_csv='language_models/laplace/bigram_perplexity_laplace_train.csv', log_prob_save_csv='language_models/laplace/bigram_log_prob_laplace_train.csv')
bigram.calc_perplexity(df_test, smoothing="laplace",
                       save_csv='language_models/laplace/bigram_perplexity_laplace_test.csv', log_prob_save_csv='language_models/laplace/bigram_log_prob_laplace_test.csv')


trigram = NGramProcessor(df_train, 3)
trigram.train()
trigram.find_probability(save_csv='language_models/trigram.csv')
trigram.calc_perplexity(df_train, save_csv='language_models/trigram_perplexity.csv')
trigram.calc_perplexity(df_train, smoothing="laplace",
                        save_csv='language_models/laplace/trigram_perplexity_laplace_train.csv', log_prob_save_csv='language_models/laplace/trigram_log_prob_laplace_train.csv')
trigram.calc_perplexity(df_test, smoothing="laplace",
                        save_csv='language_models/laplace/trigram_perplexity_laplace_test.csv', log_prob_save_csv='language_models/laplace/trigram_log_prob_laplace_test.csv')


quadgram = NGramProcessor(df_train, 4)
quadgram.train()
quadgram.find_probability(save_csv='language_models/quadgram.csv')
quadgram.calc_perplexity(df_train, save_csv='language_models/quadgram_perplexity.csv')
quadgram.calc_perplexity(df_train, smoothing="laplace",
                            save_csv='language_models/laplace/quadgram_perplexity_laplace_train.csv', log_prob_save_csv='language_models/laplace/quadgram_log_prob_laplace_train.csv')
quadgram.calc_perplexity(df_test, smoothing="laplace",
                         save_csv='language_models/laplace/quadgram_perplexity_laplace_test.csv', log_prob_save_csv='language_models/laplace/quadgram_log_prob_laplace_test.csv')
