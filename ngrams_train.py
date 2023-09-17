from NGramProcessor import *
import os

os.makedirs('language_models', exist_ok=True)
os.makedirs('average_perplexity', exist_ok=True)

df_train = pd.read_csv('dataset/train_dataset.csv')

df_test = pd.read_csv('dataset/validate_dataset.csv')

unigram = NGramProcessor(df_train, 1)
unigram.train()
# unigram.calc_perplexity(df_train, perplexity_csv='perplexities/no_smoothing/perplexity_unigram_train.csv',
#                         log_prob_save_csv='language_models/no_smoothing/unigram_train.csv')
# unigram.calc_perplexity(df_test, perplexity_csv='perplexities/no_smoothing/perplexity_unigram_test.csv',
#                         log_prob_save_csv='language_models/no_smoothing/unigram_test.csv')

# unigram.calc_perplexity(df_train, perplexity_csv='perplexities/laplace/perplexity_unigram_train.csv', smoothing='laplace',
#                         log_prob_save_csv='language_models/laplace/unigram_train.csv')
# unigram.calc_perplexity(df_test, perplexity_csv='perplexities/laplace/perplexity_unigram_test.csv', smoothing='laplace',
#                         log_prob_save_csv='language_models/laplace/unigram_test.csv')

unigram.calc_perplexity(df_train, perplexity_csv='perplexities/additive/perplexity_unigram_train.csv', smoothing='additive', k=0.01,
                        log_prob_save_csv='language_models/additive/unigram_train.csv')
unigram.calc_perplexity(df_test, perplexity_csv='perplexities/additive/perplexity_unigram_test.csv', smoothing='additive', k=0.01,
                        log_prob_save_csv='language_models/additive/unigram_test.csv')
unigram.calc_perplexity(df_train, perplexity_csv='perplexities/additive/perplexity_unigram_train.csv', smoothing='additive', k=0.1,
                        log_prob_save_csv='language_models/additive/unigram_train.csv')
unigram.calc_perplexity(df_test, perplexity_csv='perplexities/additive/perplexity_unigram_test.csv', smoothing='additive', k=0.1,
                        log_prob_save_csv='language_models/additive/unigram_test.csv')
unigram.calc_perplexity(df_train, perplexity_csv='perplexities/additive/perplexity_unigram_train.csv', smoothing='additive', k=10,
                        log_prob_save_csv='language_models/additive/unigram_train.csv')
unigram.calc_perplexity(df_test, perplexity_csv='perplexities/additive/perplexity_unigram_test.csv', smoothing='additive', k=10,
                        log_prob_save_csv='language_models/additive/unigram_test.csv')
unigram.calc_perplexity(df_train, perplexity_csv='perplexities/additive/perplexity_unigram_train.csv', smoothing='additive', k=100,
                        log_prob_save_csv='language_models/additive/unigram_train.csv')
unigram.calc_perplexity(df_test, perplexity_csv='perplexities/additive/perplexity_unigram_test.csv', smoothing='additive', k=100,
                        log_prob_save_csv='language_models/additive/unigram_test.csv')

unigram.calc_perplexity(df_train, smoothing='turing', perplexity_csv='perplexities/turing/perplexity_unigram_train.csv',
                        log_prob_save_csv='language_models/turing/unigram_train.csv')
unigram.calc_perplexity(df_test, smoothing='turing', perplexity_csv='perplexities/perplexity_turing_unigram_test.csv',
                        log_prob_save_csv='language_models/turing/unigram_test.csv')
