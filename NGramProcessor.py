import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm 
import re

class NGramProcessor:
    """
    This class is used to create n-gram language models
    """    
    def __init__(self, df_train, df_test, n=1):
        """

        Args:
            df_train (pd.DataFrame): Preprocessed training dataset
            df_test (pd.DataFrame): Preprocessed validation dataset
            ngram (int, optional): Value of n for the n-gram. Defaults to 1.
        """        
        self.df_train = df_train
        self.df_test = df_test
        self.n = n
        pass

    def make_sentences(self, comment):
        """Function to create a sentence from a comment

        Args:
            comment (str): Comment to create a sentence from

        Returns:
            list: list of sentences
        """        
        pattern = r'<s> <s> <s>(.*?)</s> </s> </s>'
        sentences = re.findall(pattern, comment, re.DOTALL)
        sentences = ['<s> <s> <s> ' + sentence.strip() + ' </s> </s> </s>' for sentence in sentences]
        return sentences

    def ngrams(self, df, n=1, startIdx=0):
        """Function to create n-grams from a dataset

        Args:
            df (pd.DataFrame): Preprocessed dataset to create n-grams from
            n (int, optional): Value of n for the n-gram. Defaults to 1.

        Returns:
            list: list of n-grams
        """        
        ngrams_corpus = []
        for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f'Creating {n}-grams'):
            sentences = self.make_sentences(row['Processed_Comment'])
            for sentence in sentences:
                words = sentence.split(" ")
                ngrams_sentence = [' '.join(words[i:i + n]) for i in range(startIdx, len(words) - n - 1)]
                ngrams_corpus.extend(ngrams_sentence)
        return ngrams_corpus

    def create_frequency_dict(self, n, startIdx=0):
        """Function to create a frequency dictionary for n-grams

        Args:
            n (int): Value of n for the n-gram

        Returns:
            Dict: Dictionary of n-grams and their frequencies
        """        
        ngrams = self.ngrams(self.df_train, n, startIdx)
        frequency_dict = defaultdict(int)
        
        for item in tqdm(ngrams, desc=f'Creating frequency dictionary for {n}-grams'):
            frequency_dict[item] += 1
            # if item in frequency_dict:
            #     frequency_dict[item] += 1
            # else:
            #     frequency_dict[item] = 1
        return frequency_dict

    def find_probability(self, save_csv='sample.csv'):
        """Function to find the probability of n-grams

        Args:
            save_csv (str, optional): Path to save the csv file. Defaults to 'sample.csv'.

        Returns:
            pd.DataFrame: Dataframe containing the n-grams and their probabilities
        """        
        if(self.n != 1):
            frequency_dict_n_1 = self.create_frequency_dict(self.n-1, 4-self.n)
        frequency_dict = self.create_frequency_dict(self.n, 4-self.n)
        probability_dict = {}

        # if train:
        for key in tqdm(frequency_dict.keys(), desc=f'Finding probability for {self.n}-grams'):
            if self.n == 1:
                probability_dict[key] = frequency_dict[key] / sum(frequency_dict.values())
            else:
                key_n_1 = ' '.join(key.split()[:self.n - 1])
                if bool(frequency_dict_n_1.get(key_n_1)):
                    probability_dict[key] = frequency_dict[key] / frequency_dict_n_1[key_n_1] 
                else:
                    probability_dict[key] = 0

        df = pd.DataFrame(list(probability_dict.items()), columns=['Comment', 'Probability'])

        df.to_csv(save_csv, index=False)
        print(f'Saved {self.n}-gram probabilities to {save_csv}')
        return df

    
    def calc_perplexity(self, save_csv='sample.csv'):
        """Function to find the probability of n-grams

        Args:
            save_csv (str, optional): Path to save the csv file. Defaults to 'sample.csv'.

        Returns:
            pd.DataFrame: Dataframe containing the n-grams and their probabilities
        """
        
        if(self.n != 1):
            frequency_dict_n_1 = self.create_frequency_dict(self.n-1, 4-self.n)
        frequency_dict = self.create_frequency_dict(self.n, 4-self.n)
        probability_dict = {}

        ngram = self.ngrams(self.df_test, self.n)
        for key in list(set(ngram)):
            if self.n == 1:
                probability_dict[key] = frequency_dict[key] / sum(frequency_dict.values())

            else:
                key_n_1 = ' '.join(key.split()[:self.n - 1])
                if bool(frequency_dict_n_1.get(key_n_1)):
                    probability_dict[key] = frequency_dict[key] / frequency_dict_n_1[key_n_1]
                else:
                    probability_dict[key] = 0
                # probability_dict[key] = (0 if not bool(frequency_dict_n_1.get(key_n_1)) else (frequency_dict[key] if bool(frequency_dict.get(key)) else 0) / frequency_dict_n_1[key_n_1])
        
        df = pd.DataFrame(columns=['Comment', 'Perplexity'])
        avg_perplexity = 0
        count_sentences = 0
        for idx, row in tqdm(self.df_test.iterrows(), total=self.df_test.shape[0], desc=f'Calculating perplexity for {self.n}-grams'):
            sentences = self.make_sentences(row['Processed_Comment'])
            perplexity = 0
            for sentence in sentences:
                words = sentence.split(" ")
                ngrams_sentence = [' '.join(words[i:i + self.n]) for i in range(4-self.n, len(words) - self.n - 1)]
                total_log_prob = 0
                for ngram_set in ngrams_sentence:
                    total_log_prob += np.log2(probability_dict[ngram_set])
                perplexity += 2 ** (-total_log_prob / len(ngrams_sentence))
            
            # df.at[idx, 'Perplexity'] = perplexity/len(sentences)
            df.loc[idx] = [row['Processed_Comment'], perplexity/len(sentences)]
            avg_perplexity += perplexity
            count_sentences += len(sentences)
        avg_perplexity /= count_sentences
        print(f'Average perplexity for {self.n}-grams: {avg_perplexity}')
        perp_df = pd.read_csv('avg_perplexity.csv').loc[self.n-1, 'Average Perplexity'] = avg_perplexity
        perp_df.to_csv('avg_perplexity.csv', index=False)

        df.to_csv(save_csv, index=False)
        print(f'Saved {self.n}-gram perplexities to {save_csv}')
        return df