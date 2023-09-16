import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from tqdm import tqdm
import multiprocessing
from joblib import Parallel, delayed

tqdm.pandas()

# nltk.download('punkt')
# nltk.download('stopwords')

df = pd.read_csv('raw_reddit_data_filtered.csv')
# Create a multiprocessing manager to create a shared list
# manager = multiprocessing.Manager()
# sentence_list = manager.list()

def preprocess_comment(comment):
    # global sentence_list
    link_pattern = r'http\S+|www\S+|\@\w+|\#'
    special_character_pattern = r'[^A-Za-z0-9\s]'
    
    sentences = sent_tokenize(comment)
    
    processed_sentences = []
    
    for sentence in sentences:
        sentence = re.sub(link_pattern, '', sentence)
        sentence = re.sub(special_character_pattern, '', sentence)
        sentence = sentence.replace('x000D', '')
        sentence = sentence.strip()

        # words = word_tokenize(sentence)
        
        # filtered_words = [word.lower() for word in words if word.lower()]
     
        # processed_sentence = ' '.join(filtered_words)

        # if (len(re.sub(' ', '', processed_sentence)) == 0):
            # continue
        
        processed_sentence = '<s> <s> <s> ' + sentence + ' </s> </s> </s>'
        
        # sentence_list.append(processed_sentence)
        processed_sentences.append(processed_sentence)
    
    # processed_comment = ' '.join(processed_sentences)
    
    return processed_sentences



# Use Parallel and delayed to parallelize the processing of comments
num_cores = multiprocessing.cpu_count()
processed_comments = Parallel(n_jobs=num_cores)(delayed(preprocess_comment)(comment) for comment in tqdm(df['Comment']))
# df['Processed_Comment'] = processed_comments
# df = df[df['Processed_Comment'].str.strip() != '']
# df.to_csv('processed_dataset.csv', index=False)


processed_comments = [item for sublist in processed_comments for item in sublist]
sentence_df = pd.DataFrame(processed_comments, columns=['Sentences'])
sentence_df.to_csv('sentences.csv', index=False)
print('Saved sentences to sentences.csv')