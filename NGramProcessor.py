import pandas as pd

class NGramProcessor:
    """
    This class is used to create n-gram language models
    """    
    def __init__(self, df_train, df_test, ngram=1):
        """

        Args:
            df_train (pd.DataFrame): Preprocessed training dataset
            df_test (pd.DataFrame): Preprocessed validation dataset
            ngram (int, optional): Value of n for the n-gram. Defaults to 1.
        """        
        self.df_train = df_train
        self.df_test = df_test
        self.ngram = ngram
        pass

    def ngrams(self, df, ngram=1):
        """Function to create n-grams from a dataset

        Args:
            df (pd.DataFrame): Preprocessed dataset to create n-grams from
            ngram (int, optional): Value of n for the n-gram. Defaults to 1.

        Returns:
            list: list of n-grams
        """        
        ngrams_corpus = []
        for _, row in df.iterrows():
            words = row['Processed_Comment'].split(" ")
            ngrams_sentence = [' '.join(words[i:i + ngram]) for i in range(len(words) - ngram + 1)]
            ngrams_corpus.extend(ngrams_sentence)
        return ngrams_corpus

    def create_frequency_dict(self, ngram):
        """Function to create a frequency dictionary for n-grams

        Args:
            ngram (int): Value of n for the n-gram

        Returns:
            Dict: Dictionary of n-grams and their frequencies
        """        
        ngram = self.ngrams(self.df_train, ngram)
        frequency_dict = {}
        for item in ngram:
            if item in frequency_dict:
                frequency_dict[item] += 1
            else:
                frequency_dict[item] = 1
        return frequency_dict

    def find_probability(self, train=True, save_csv=''):
        """Function to find the probability of n-grams

        Args:
            train (bool, optional): Boolean value to indicate if the function is being called for 
                                    training or validation. Defaults to True.
            save_csv (str, optional): Path to save the csv file. Defaults to ''.

        Returns:
            pd.DataFrame: Dataframe containing the n-grams and their probabilities
        """        
        if(self.ngram != 1):
            frequency_dict_n_1 = self.create_frequency_dict(self.ngram - 1)
        frequency_dict = self.create_frequency_dict(self.ngram)
        probability_dict = {}
        if train:
            for key in frequency_dict.keys():
                if self.ngram == 1:
                    probability_dict[key] = (frequency_dict[key] if bool(frequency_dict.get(key)) else 0) / sum(frequency_dict.values())
                else:
                    key_n_1 = ' '.join(key.split()[:self.ngram - 1])
                    probability_dict[key] = (frequency_dict[key] if bool(frequency_dict.get(key)) else 0) / (frequency_dict_n_1[key_n_1] if bool(frequency_dict_n_1.get(key_n_1)) else 0)
        else:
            ngram = self.ngrams(self.df_test, self.ngram)
            for key in list(set(ngram)):
                if self.ngram == 1:
                    probability_dict[key] = (frequency_dict[key] if bool(frequency_dict.get(key)) else 0) / sum(frequency_dict.values())

                else:
                    key_n_1 = ' '.join(key.split()[:self.ngram - 1])
                    probability_dict[key] = (0 if not bool(frequency_dict_n_1.get(key_n_1)) else (frequency_dict[key] if bool(frequency_dict.get(key)) else 0) / frequency_dict_n_1[key_n_1])

        if save_csv != '':
            df = pd.DataFrame(list(probability_dict.items()), columns=['Phrase', 'Probability'])
            df.to_csv(save_csv, index=False)
        return df
