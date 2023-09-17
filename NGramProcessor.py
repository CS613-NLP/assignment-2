import pandas as pd
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from joblib import Parallel, delayed
import os

tqdm.pandas()


class NGramProcessor:
    """
    This class is used to create n-gram language models
    """

    def __init__(self, df_train, n=1):
        """

        Args:
            df_train (pd.DataFrame): Preprocessed training dataset
            ngram (int, optional): Value of n for the n-gram. Defaults to 1.
        """
        self.n = n
        self.df_train = df_train
        pass

    def ngrams(self, comment, n=1):
        """Function to create n-grams from a comment

        Args:
            comment (str): Comment from the dataset
            n (int, optional): Value of n for the n-gram. Defaults to 1.

        Returns:
            List: List of n-grams from the comment
        """
        words = comment.split()
        # for all the words also split based on \n or \t
        all_words = []
        for word in words:
            all_words.extend(word.split('\n'))

        return [' '.join(all_words[i:i + n]) for i in range(4-self.n, len(all_words) - n - 1)]

    def create_frequency_dict(self, n):
        """Function to create a frequency dictionary for n-grams

        Args:
            n (int): Value of n for the n-gram

        Returns:
            Dict: Dictionary of n-grams and their frequencies
        """
        # ngrams = self.ngrams(self.df_train, n, startIdx)
        ngrams_series = self.df_train['Sentences'].progress_apply(
            self.ngrams, args=(n,))
        ngrams = ngrams_series.explode().tolist()
        frequency_dict = defaultdict(int)

        for item in tqdm(ngrams, desc=f'Creating frequency dictionary for {n}-grams'):
            frequency_dict[item] += 1

        return frequency_dict

    def train(self):
        """Function to train the n-gram model
        """
        self.vocab_size = self.__get_vocab_size()
        self.frequency_dict = self.create_frequency_dict(self.n)
        self.frequency_dict_size = sum(self.frequency_dict.values())
        if(self.n != 1):
            self.frequency_dict_n_1 = self.create_frequency_dict(self.n-1)
    
    def compute_log_probability(self, key, smoothing=None):
        """Function to compute the log probability of an n-gram

        Args:
            key (str): n-gram

        Returns:
            Tuple: Tuple containing the n-gram and its log probability
        """

        if smoothing == "laplace" or smoothing == "additive":
            if self.n == 1:
                if bool(self.laplace_frequency_dict.get(key)):
                    return key, np.log2(self.laplace_frequency_dict[key]) - np.log2(sum(self.frequency_dict.values()))
                else:
                    return key, self.unseen_probability
            else:
                key_n_1 = ' '.join(key.split()[:self.n - 1])
                if bool(self.frequency_dict_n_1.get(key_n_1)):
                    if bool(self.laplace_frequency_dict.get(key)):
                        return key, np.log2(self.laplace_frequency_dict[key]) - np.log2(self.frequency_dict_n_1[key_n_1])
                    else:
                        return key, np.log2(self.k) - np.log2(self.frequency_dict_n_1[key_n_1] + self.k*self.vocab_size)
                else:
                    return key, np.log2(self.k) - np.log2(self.k*self.vocab_size)


        elif smoothing == "turing":
            updated_frequency_dict = self.turing_frequency_dict
            unseen_words_probability = np.log2(
                self.frequency_of_frequency[1] / self.frequency_dict_size)

            if self.n == 1:
                if bool(updated_frequency_dict.get(key)):
                    return key, np.log2(updated_frequency_dict[key]) - np.log2(sum(self.frequency_dict.values()))
                else:
                    return key, unseen_words_probability
            else:
                key_n_1 = ' '.join(key.split()[:self.n - 1])
                if bool(updated_frequency_dict.get(key)):
                    return key, np.log2(updated_frequency_dict[key]) - np.log2(self.frequency_dict_n_1[key_n_1])
                else:
                    return key, unseen_words_probability

        elif smoothing is None:
            if self.n == 1:
                if bool(self.frequency_dict.get(key)):
                    return key, np.log2(self.frequency_dict[key]) - np.log2(sum(self.frequency_dict.values()))
                else:
                    return key, -np.inf
            else:
                key_n_1 = ' '.join(key.split()[:self.n - 1])
                if bool(self.frequency_dict.get(key)):
                    return key, np.log2(self.frequency_dict[key]) - np.log2(self.frequency_dict_n_1[key_n_1])
                else:
                    return key, -np.inf

    def find_probability(self, save_csv='sample.csv'):
        """Function to find the log probability of n-grams

        Args:
            save_csv (str, optional): Path to save the csv file. Defaults to 'sample.csv'.

        Returns:
            pd.DataFrame: Dataframe containing the n-grams and their log probabilities
        """
        keys = list(self.frequency_dict.keys())

        probability_results = []
        for key in tqdm(keys, desc=f'Finding log probability for {self.n}-grams'):
            probability_results.append(self.compute_log_probability(key))

        log_probability_dict = dict(probability_results)

        df = pd.DataFrame(list(log_probability_dict.items()),
                          columns=['Comment', 'Probability'])

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

    def calc_perplexity(self, df_test, smoothing=None, save_csv='sample.csv', log_prob_save_csv='logprob.csv', k=1):
        """Function to find the probability of n-grams

        Args:
            save_csv (str, optional): Path to save the csv file. Defaults to 'sample.csv'.

        Returns:
            pd.DataFrame: Dataframe containing the n-grams and their probabilities
        """
        ngrams_series = df_test['Sentences'].progress_apply(
            self.ngrams, args=(self.n,))
        ngrams = ngrams_series.explode().tolist()

        self.k = k
        save_csv_dir_path = save_csv[:-len(save_csv.split('/')[-1])-1]
        if save_csv_dir_path != '':
            os.makedirs(save_csv_dir_path, exist_ok=True)

        # num_cores = multiprocessing.cpu_count()
        # probability_results = Parallel(n_jobs=num_cores)(
        #     delayed(self.compute_probability)(key) for key in tqdm(list(set(ngrams)), desc=f'Finding probability for {self.n}-grams')
        # )

        probability_results = []

        if smoothing == "turing":
            self.__populate_turning()
        elif smoothing == "laplace":
            self.__laplace_smoothing()
        elif smoothing == "additive":
            self.__laplace_smoothing(k)
        elif smoothing is None:
            log_prob_save_csv = None

        for key in tqdm(list(set(ngrams)), desc=f'Finding log probability for {self.n}-grams with {smoothing} smoothing'):
            probability_results.append(
                self.compute_log_probability(key, smoothing=smoothing))

        log_probability_dict = dict(probability_results)

        # save the log probability dict
        if log_prob_save_csv is not None:
            log_prob_save_csv_dir_path = log_prob_save_csv[:-len(log_prob_save_csv.split('/')[-1])-1]
            if log_prob_save_csv_dir_path != '':
                os.makedirs(log_prob_save_csv_dir_path, exist_ok=True)
            df = pd.DataFrame(list(log_probability_dict.items()), columns=['Comment', 'Log Probability'])
            df.to_csv(log_prob_save_csv, index=False)
            print(f'Saved {self.n}-gram probabilities to {log_prob_save_csv}')

        sentences = df_test['Sentences'].tolist()

        perplexity_scores = []
        for sentence in tqdm(sentences, desc=f'Calculating perplexity for {self.n}-grams'):
            perplexity_scores.append(self.calculate_sentence_perplexity(
                sentence, log_probability_dict))

        df = pd.DataFrame(
            {'Comment': sentences, 'Perplexity': perplexity_scores})
        avg_perplexity = np.mean(perplexity_scores)

        print(
            f'Average perplexity for {self.n}-grams: {avg_perplexity} with {smoothing} smoothing')


        if df_test.equals(self.df_train):
            avg_file = f'average_perplexity/avg_perplexity_{smoothing}_smoothing_train.csv'
        else:
            avg_file = f'average_perplexity/avg_perplexity_{smoothing}_smoothing_test.csv'

        # if avg file does not exist, create it
        if not os.path.exists(avg_file):
            temp_df = pd.DataFrame(
                {'n': ["Unigram", "Bigram", "Trigram", "Quadgram"], 'Average Perplexity': [0, 0, 0, 0]})
            temp_df.to_csv(avg_file, index=False)

        perp_df = pd.read_csv(avg_file)
        perp_df.loc[self.n-1, 'Average Perplexity'] = avg_perplexity
        perp_df.to_csv(avg_file, index=False)

        df.to_csv(save_csv, index=False)
        print(
            f'Saved {self.n}-gram perplexities with {smoothing} smoothing to {save_csv}')
        return df

    def __populate_turning(self):
        self.frequency_of_frequency = {}

        for key in self.frequency_dict.keys():
            if self.frequency_dict[key] not in self.frequency_of_frequency:
                self.frequency_of_frequency[self.frequency_dict[key]] = 1
            else:
                self.frequency_of_frequency[self.frequency_dict[key]] += 1

        self.frequency_of_frequency[0] = (
            self.vocab_size ** (self.n)) - len(self.frequency_dict.keys())

        self.turing_frequency_dict = {}

        for key in self.frequency_dict.keys():
            count = self.frequency_dict[key]
            freq_count = self.frequency_of_frequency[count]

            if count + 1 in self.frequency_of_frequency:
                freq_count_plus = self.frequency_of_frequency[count + 1]
            else:
                freq_count_plus = 0

            self.turing_frequency_dict[key] = (
                (count + 1) * freq_count_plus) / freq_count

    def __laplace_smoothing(self, k=1):
        """
        This applies laplace smoothing to the n-grams
        """
        self.vocab_size = self.__get_vocab_size()
        self.laplace_frequency_dict = {}
        N = sum(self.frequency_dict.values())

        # if train:
        for key in tqdm(self.frequency_dict.keys(), desc=f'Finding probability for {self.n}-grams'):
            if self.n == 1:
                self.laplace_frequency_dict[key] = (
                    (self.frequency_dict[key] + k*1)*N) / (N + k*self.vocab_size)
            else:
                key_n_1 = ' '.join(key.split()[:self.n - 1])
                self.laplace_frequency_dict[key] = ((self.frequency_dict[key] + k*1)*self.frequency_dict_n_1[key_n_1]) / (
                    self.frequency_dict_n_1[key_n_1] + k*self.vocab_size)

        if self.n == 1:
            self.unseen_probability = np.log2(k) - np.log2(N + k*self.vocab_size)
        else:
            self.unseen_probability = np.log2(k) - np.log2(self.frequency_dict_n_1[key_n_1] + k*self.vocab_size)


    def __get_vocab_size(self):
        words_set = set()
        for sentence in self.df_train['Sentences']:
            words = sentence.split()
            for word in words:
                words_set.add(word)

        self.vocab_size = len(words_set)-2
        return len(words_set)-2
