from NGramProcessor import *
import os

os.makedirs('language_models', exist_ok=True)
# os.makedirs('language_models/laplace', exist_ok=True)

df_train = pd.read_csv('dataset/train_dataset.csv')

df_test = pd.read_csv('dataset/validate_dataset.csv')

unigram = NGramProcessor(df_train, 1)
unigram.train()
# unigram.calc_perplexity(df_train, perplexity_csv='perplexities/perplexity_no_smoothing_unigram_train.csv',
#                         log_prob_save_csv='language_models/no_smoothing/unigram_train.csv')
# unigram.calc_perplexity(df_test, perplexity_csv='perplexities/perplexity_no_smoothing_unigram_test.csv',
#                         log_prob_save_csv='language_models/no_smoothing/unigram_test.csv')
unigram.calc_perplexity(df_train, perplexity_csv='perplexities/perplexity_laplace_unigram_train.csv', smoothing='laplace',
                        log_prob_save_csv='language_models/laplace/unigram_train.csv')
# unigram.calc_perplexity(df_train, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_unigram_train.csv',
#                         log_prob_save_csv='language_models/unigram/turing_unigram_train.csv')
# unigram.calc_perplexity(df_test, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_unigram_test.csv',
#                         log_prob_save_csv='language_models/unigram/turing_unigram_test.csv')

bigram = NGramProcessor(df_train, 2)
bigram.train()
# bigram.calc_perplexity(df_train, perplexity_csv='perplexities/perplexity_no_smoothing_bigram_train.csv',
#                        log_prob_save_csv='language_models/no_smoothing/bigram_train.csv')
# bigram.calc_perplexity(df_test, perplexity_csv='perplexities/perplexity_no_smoothing_bigram_test.csv',
#                        log_prob_save_csv='language_models/no_smoothing/bigram_test.csv')
# # bigram.calc_perplexity(df_train, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_bigram_train.csv',
# #                        log_prob_save_csv='language_models/bigram/turing_bigram_train.csv')
# # bigram.calc_perplexity(df_test, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_bigram_test.csv',
# #                        log_prob_save_csv='language_models/bigram/turing_bigram_test.csv')
bigram.calc_perplexity(df_train, perplexity_csv='perplexities/perplexity_laplace_bigram_train.csv', smoothing='laplace',
                       log_prob_save_csv='language_models/laplace/bigram_train.csv')

trigram = NGramProcessor(df_train, 3)
trigram.train()
# trigram.calc_perplexity(df_train, perplexity_csv='perplexities/perplexity_no_smoothing_trigram_train.csv',
#                         log_prob_save_csv='language_models/no_smoothing/trigram_train.csv')
# trigram.calc_perplexity(df_test, perplexity_csv='perplexities/perplexity_no_smoothing_trigram_test.csv',
#                         log_prob_save_csv='language_models/no_smoothing/trigram_test.csv')
# # trigram.calc_perplexity(df_train, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_bigram_train.csv',
# #                         log_prob_save_csv='language_models/bigram/turing_bigram_train.csv')
# # trigram.calc_perplexity(df_test, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_bigram_test.csv',
# #                         log_prob_save_csv='language_models/bigram/turing_bigram_test.csv')
trigram.calc_perplexity(df_train, perplexity_csv='perplexities/perplexity_laplace_trigram_train.csv', smoothing='laplace',
                        log_prob_save_csv='language_models/laplace/trigram_train.csv')

quadgram = NGramProcessor(df_train, 4)
quadgram.train()
# quadgram.calc_perplexity(df_train, perplexity_csv='perplexities/perplexity_no_smoothing_quadgram_train.csv',
#                          log_prob_save_csv='language_models/no_smoothing/quadgram_train.csv')
# quadgram.calc_perplexity(df_test, perplexity_csv='perplexities/perplexity_no_smoothing_quadgram_test.csv',
#                          log_prob_save_csv='language_models/no_smoothing/quadgram_test.csv')
# # quadgram.calc_perplexity(df_train, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_quadgram_train.csv',
# #                          log_prob_save_csv='language_models/quadgram/turing_quadgram_train.csv')
# # quadgram.calc_perplexity(df_test, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_quadgram_test.csv',
# #                          log_prob_save_csv='language_models/quadgram/turing_quadgram_test.csv')
quadgram.calc_perplexity(df_train, perplexity_csv='perplexities/perplexity_laplace_quadgram_train.csv', smoothing='laplace',
                         log_prob_save_csv='language_models/laplace/quadgram_train.csv')
