from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords

def get_tf_idf(lyrics):

    stop_words = set(stopwords.words('english'))

    filtered_lyrics = []
    for lyric in lyrics:
        words = lyric.split(' ')
        filtered = [w for w in words if not w.lower() in stop_words]
        filtered_lyrics.append(" ".join(filtered))

    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(filtered_lyrics)

    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(word_count_vector)

    tf_idf = pd.DataFrame(X.toarray() ,columns=cv.get_feature_names_out())

    tf_idf_per_song = tf_idf.sum(axis=1)
    return tf_idf_per_song
