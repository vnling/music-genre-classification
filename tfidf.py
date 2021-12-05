from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import re
import pandas as pd
import nltk
import numpy as np


def get_tf_idf(file):
    df = pd.read_csv(file)
    df.dropna()

    tokenizer = nltk.RegexpTokenizer(r"\w+")
    for idx, row in df.iterrows():
        lyrics = df.loc[idx,'lyrics']
        clean_lyrics = tokenizer.tokenize(lyrics)
        clean_lyrics = " ".join(clean_lyrics)
        df.loc[idx,'lyrics'] = clean_lyrics

    lyrics = df['lyrics']
            
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(lyrics)

    tf = pd.DataFrame(word_count_vector.toarray(), columns=cv.get_feature_names_out())

    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(word_count_vector)
    idf = pd.DataFrame({'feature_name':cv.get_feature_names_out(), 'idf_weights':tfidf_transformer.idf_})

    tf_idf = pd.DataFrame(X.toarray() ,columns=cv.get_feature_names_out())

    tf_idf_per_song = tf_idf.sum(axis=1)
    return np.array(tf_idf_per_song)
