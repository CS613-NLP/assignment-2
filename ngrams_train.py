from NGramProcessor import *
import os

os.makedirs('language_models', exist_ok=True)
os.makedirs('average_perplexity', exist_ok=True)

df_train = pd.read_csv('dataset/train_dataset.csv')

df_test = pd.read_csv('dataset/validate_dataset.csv')

quadgram = NGramProcessor(df_train, 4)
quadgram.train()
quadgram.calc_perplexity(df_train, perplexity_csv='perplexities/no_smoothing/perplexity_quadgram_train.csv',
                         log_prob_save_csv='language_models/no_smoothing/quadgram_train.csv')
quadgram.calc_perplexity(df_test, perplexity_csv='perplexities/no_smoothing/perplexity_quadgram_test.csv',
                         log_prob_save_csv='language_models/no_smoothing/quadgram_test.csv')

quadgram.calc_perplexity(df_train, perplexity_csv='perplexities/laplace/perplexity_quadgram_train.csv', smoothing='laplace',
                         log_prob_save_csv='language_models/laplace/quadgram_train.csv')
quadgram.calc_perplexity(df_test, perplexity_csv='perplexities/laplace/perplexity_quadgram_test.csv', smoothing='laplace',
                         log_prob_save_csv='language_models/laplace/quadgram_test.csv')

quadgram.calc_perplexity(df_train, perplexity_csv='perplexities/additive/perplexity_quadgram_train.csv', smoothing='additive', k=0.01,
                         log_prob_save_csv='language_models/additive/quadgram_train.csv')
quadgram.calc_perplexity(df_test, perplexity_csv='perplexities/additive/perplexity_quadgram_test.csv', smoothing='additive', k=0.01,
                         log_prob_save_csv='language_models/additive/quadgram_test.csv')
quadgram.calc_perplexity(df_train, perplexity_csv='perplexities/additive/perplexity_quadgram_train.csv', smoothing='additive', k=0.1,
                         log_prob_save_csv='language_models/additive/quadgram_train.csv')
quadgram.calc_perplexity(df_test, perplexity_csv='perplexities/additive/perplexity_quadgram_test.csv', smoothing='additive', k=0.1,
                         log_prob_save_csv='language_models/additive/quadgram_test.csv')
quadgram.calc_perplexity(df_train, perplexity_csv='perplexities/additive/perplexity_quadgram_train.csv', smoothing='additive', k=10,
                         log_prob_save_csv='language_models/additive/quadgram_train.csv')
quadgram.calc_perplexity(df_test, perplexity_csv='perplexities/additive/perplexity_quadgram_test.csv', smoothing='additive', k=10,
                         log_prob_save_csv='language_models/additive/quadgram_test.csv')
quadgram.calc_perplexity(df_train, perplexity_csv='perplexities/additive/perplexity_quadgram_train.csv', smoothing='additive', k=100,
                         log_prob_save_csv='language_models/additive/quadgram_train.csv')
quadgram.calc_perplexity(df_test, perplexity_csv='perplexities/additive/perplexity_quadgram_test.csv', smoothing='additive', k=100,
                         log_prob_save_csv='language_models/additive/quadgram_test.csv')

quadgram.calc_perplexity(df_train, smoothing='turing', perplexity_csv='perplexities/turing/perplexity_quadgram_train.csv',
                         log_prob_save_csv='language_models/turing/quadgram_train.csv')
quadgram.calc_perplexity(df_test, smoothing='turing', perplexity_csv='perplexities/turing/perplexity_quadgram_test.csv',
                         log_prob_save_csv='language_models/turing/quadgram_test.csv')
