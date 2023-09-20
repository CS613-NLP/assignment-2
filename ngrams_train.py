from NGramProcessor import *
import os

# Create folders to save the results
os.makedirs('language_models', exist_ok=True)
os.makedirs('average_perplexity', exist_ok=True)

# Get the preprocessed dataset
df_train = pd.read_csv('dataset/train_dataset.csv')
df_test = pd.read_csv('dataset/validate_dataset.csv')


# Input the n-gram you want to train. 1 for unigram, 2 for bigram, 3 for trigram, 4 for quadgram
n = 4

if n==1:
    n_gram = "unigram"
elif n==2:
    n_gram = "bigram"
elif n==3:
    n_gram = "trigram"
elif n==4:
    n_gram = "quadgram"
else:
    n_gram = f"{n}-gram"


ngram = NGramProcessor(df_train, 4)
ngram.train()
ngram.calc_perplexity(df_train, perplexity_csv=f'perplexities/no_smoothing/perplexity_{n_gram}_train.csv',
                         log_prob_save_csv=f'language_models/no_smoothing/{n_gram}_train.csv')
ngram.calc_perplexity(df_test, perplexity_csv=f'perplexities/no_smoothing/perplexity_{n_gram}_test.csv',
                         log_prob_save_csv=f'language_models/no_smoothing/{n_gram}_test.csv')

ngram.calc_perplexity(df_train, perplexity_csv=f'perplexities/laplace/perplexity_{n_gram}_train.csv', smoothing='laplace',
                         log_prob_save_csv=f'language_models/laplace/{n_gram}_train.csv')
ngram.calc_perplexity(df_test, perplexity_csv=f'perplexities/laplace/perplexity_{n_gram}_test.csv', smoothing='laplace',
                         log_prob_save_csv=f'language_models/laplace/{n_gram}_test.csv')

ngram.calc_perplexity(df_train, perplexity_csv=f'perplexities/additive/perplexity_{n_gram}_train.csv', smoothing='additive', k=0.01,
                         log_prob_save_csv=f'language_models/additive/{n_gram}_train.csv')
ngram.calc_perplexity(df_test, perplexity_csv=f'perplexities/additive/perplexity_{n_gram}_test.csv', smoothing='additive', k=0.01,
                         log_prob_save_csv=f'language_models/additive/{n_gram}_test.csv')
ngram.calc_perplexity(df_train, perplexity_csv=f'perplexities/additive/perplexity_{n_gram}_train.csv', smoothing='additive', k=0.1,
                         log_prob_save_csv=f'language_models/additive/{n_gram}_train.csv')
ngram.calc_perplexity(df_test, perplexity_csv=f'perplexities/additive/perplexity_{n_gram}_test.csv', smoothing='additive', k=0.1,
                         log_prob_save_csv=f'language_models/additive/{n_gram}_test.csv')
ngram.calc_perplexity(df_train, perplexity_csv=f'perplexities/additive/perplexity_{n_gram}_train.csv', smoothing='additive', k=10,
                         log_prob_save_csv=f'language_models/additive/{n_gram}_train.csv')
ngram.calc_perplexity(df_test, perplexity_csv=f'perplexities/additive/perplexity_{n_gram}_test.csv', smoothing='additive', k=10,
                         log_prob_save_csv=f'language_models/additive/{n_gram}_test.csv')
ngram.calc_perplexity(df_train, perplexity_csv=f'perplexities/additive/perplexity_{n_gram}_train.csv', smoothing='additive', k=100,
                         log_prob_save_csv=f'language_models/additive/{n_gram}_train.csv')
ngram.calc_perplexity(df_test, perplexity_csv=f'perplexities/additive/perplexity_{n_gram}_test.csv', smoothing='additive', k=100,
                         log_prob_save_csv=f'language_models/additive/{n_gram}_test.csv')

ngram.calc_perplexity(df_train, smoothing='turing', perplexity_csv=f'perplexities/turing/perplexity_{n_gram}_train.csv',
                         log_prob_save_csv=f'language_models/turing/{n_gram}_train.csv')
ngram.calc_perplexity(df_test, smoothing='turing', perplexity_csv=f'perplexities/turing/perplexity_{n_gram}_test.csv',
                         log_prob_save_csv=f'language_models/turing/{n_gram}_test.csv')
