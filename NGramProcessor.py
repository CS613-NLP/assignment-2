import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm 
import multiprocessing
from joblib import Parallel, delayed

tqdm.pandas()

class NGramProcessor:
    """
    This class is used to create n-gram language models
    """    
    def __init__(self, df_train, n=1):
        """

        Args:
            df_train (pd.DataFrame): Preprocessed training dataset
            df_test (pd.DataFrame): Preprocessed validation dataset
            ngram (int, optional): Value of n for the n-gram. Defaults to 1.
        """
        self.n = n
        self.df_train = df_train
        pass

    def ngrams(self, comment, n=1):
        """Function to create n-grams from a dataset

        Args:
            df (pd.DataFrame): Preprocessed dataset to create n-grams from
            n (int, optional): Value of n for the n-gram. Defaults to 1.

        Returns:
            list: list of n-grams
        """
        words = comment.split(" ")
        return [' '.join(words[i:i + n]) for i in range(4-self.n, len(words) - n - 1)]

    def create_frequency_dict(self, n):
        """Function to create a frequency dictionary for n-grams

        Args:
            n (int): Value of n for the n-gram

        Returns:
            Dict: Dictionary of n-grams and their frequencies
        """
        # ngrams = self.ngrams(self.df_train, n, startIdx)
        ngrams_series = self.df_train['Sentences'].progress_apply(self.ngrams, args=(n,))
        ngrams = ngrams_series.explode().tolist()
        frequency_dict = defaultdict(int)
        
        for item in tqdm(ngrams, desc=f'Creating frequency dictionary for {n}-grams'):
            frequency_dict[item] += 1

        return frequency_dict
    
    def train(self):
        """Function to train the model
        """
        self.frequency_dict = self.create_frequency_dict(self.n)
        if(self.n != 1):
            self.frequency_dict_n_1 = self.create_frequency_dict(self.n-1)

    def compute_log_probability(self, key):
        if self.n == 1:
            return key, np.log2(self.frequency_dict[key]) - np.log2(sum(self.frequency_dict.values()))
        else:
            key_n_1 = ' '.join(key.split()[:self.n - 1])
            if bool(self.frequency_dict_n_1.get(key_n_1)):
                return key, np.log2(self.frequency_dict[key]) - np.log2(self.frequency_dict_n_1[key_n_1])
            else:
                return key, 0

    def find_probability(self, save_csv='sample.csv'):
        """Function to find the log probability of n-grams

        Args:
            save_csv (str, optional): Path to save the csv file. Defaults to 'sample.csv'.

        Returns:
            pd.DataFrame: Dataframe containing the n-grams and their log probabilities
        """
        keys = list(self.frequency_dict.keys())

        # num_cores = multiprocessing.cpu_count()
        # probability_results = Parallel(n_jobs=num_cores)(
        #     delayed(self.compute_log_probability)(key) for key in tqdm(keys, desc=f'Finding log probability for {self.n}-grams')
        # )

        # log_probability_dict = dict(probability_results)

        probability_results = []
        for key in tqdm(keys, desc=f'Finding log probability for {self.n}-grams'):
            probability_results.append(self.compute_log_probability(key))

        log_probability_dict = dict(probability_results)

        df = pd.DataFrame(list(log_probability_dict.items()), columns=['Comment', 'Probability'])

        df.to_csv(save_csv, index=False)
        print(f'Saved {self.n}-gram probabilities to {save_csv}')
        return df

    def calculate_sentence_perplexity(self, sentence, log_probability_dict):
        ngrams_of_sentence = self.ngrams(sentence, self.n)
        total_log_prob = 0
        for ngram_set in ngrams_of_sentence:
            total_log_prob += log_probability_dict[ngram_set]
        perplexity = 2 ** (-total_log_prob / len(ngrams_of_sentence))
        return perplexity
    
    def calc_perplexity(self, df_test, save_csv='sample.csv'):
        """Function to find the probability of n-grams

        Args:
            save_csv (str, optional): Path to save the csv file. Defaults to 'sample.csv'.

        Returns:
            pd.DataFrame: Dataframe containing the n-grams and their probabilities
        """
        
        ngrams_series = df_test['Sentences'].progress_apply(self.ngrams, args=(self.n,))
        ngrams = ngrams_series.explode().tolist()

        # num_cores = multiprocessing.cpu_count()
        # probability_results = Parallel(n_jobs=num_cores)(
        #     delayed(self.compute_probability)(key) for key in tqdm(list(set(ngrams)), desc=f'Finding probability for {self.n}-grams')
        # )

        probability_results = []
        for key in tqdm(list(set(ngrams)), desc=f'Finding log probability for {self.n}-grams'):
            probability_results.append(self.compute_log_probability(key))
        
        log_probability_dict = dict(probability_results)

        # probability_dict[key] = (0 if not bool(frequency_dict_n_1.get(key_n_1)) else (frequency_dict[key] if bool(frequency_dict.get(key)) else 0) / frequency_dict_n_1[key_n_1])
        
        sentences = df_test['Sentences'].tolist()

        # num_cores = multiprocessing.cpu_count()
        # perplexity_scores = Parallel(n_jobs=num_cores)(
        #     delayed(self.calculate_sentence_perplexity)(sentence, log_probability_dict) 
        #     for sentence in tqdm(sentences, desc=f'Calculating perplexity for {self.n}-grams')
        # )

        perplexity_scores = []
        for sentence in tqdm(sentences, desc=f'Calculating perplexity for {self.n}-grams'):
            perplexity_scores.append(self.calculate_sentence_perplexity(sentence, log_probability_dict))

        df = pd.DataFrame({'Comment': sentences, 'Perplexity': perplexity_scores})
        avg_perplexity = np.mean(perplexity_scores)

        print(f'Average perplexity for {self.n}-grams: {avg_perplexity}')
        perp_df = pd.read_csv('avg_perplexity.csv')
        perp_df.loc[self.n-1, 'Average Perplexity'] = avg_perplexity
        perp_df.to_csv('avg_perplexity.csv', index=False)

        df.to_csv(save_csv, index=False)
        print(f'Saved {self.n}-gram perplexities to {save_csv}')
        return df