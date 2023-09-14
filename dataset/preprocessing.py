import pandas as pd
import nltk
import re
from nltk.tokenize import sent_tokenize, word_tokenize

# nltk.download('punkt')
# nltk.download('stopwords')

df = pd.read_csv('raw_reddit_data_filtered.csv')

def preprocess_comment(comment):
    
    link_pattern = r'http\S+|www\S+|\@\w+|\#'
    special_character_pattern = r'[^A-Za-z0-9\s]'
    
    sentences = sent_tokenize(comment)
    
    processed_sentences = []
    
    for sentence in sentences:
        sentence = re.sub(link_pattern, '', sentence)
        sentence = re.sub(special_character_pattern, '', sentence)
        sentence = sentence.replace('x000D', '')

        words = word_tokenize(sentence)
        
        filtered_words = [word.lower() for word in words if word.lower()]
     
        processed_sentence = ' '.join(filtered_words)

        if (len(re.sub(' ', '', processed_sentence)) == 0):
            continue
        
        processed_sentence = '<s> <s> <s> ' + processed_sentence + ' </s> </s> </s>'
        
        processed_sentences.append(processed_sentence)
    
    processed_comment = ' '.join(processed_sentences)
    
    return processed_comment

df['Processed_Comment'] = df['Comment'].apply(preprocess_comment)
df = df[df['Processed_Comment'].str.strip() != '']
df.to_csv('processed_dataset.csv', index=False)