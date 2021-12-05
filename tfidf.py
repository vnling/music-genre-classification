from collections import defaultdict
from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

def get_tf_idf(lyrics):
            
    cv = CountVectorizer()
    word_count_vector = cv.fit_transform(lyrics)

    tf = pd.DataFrame(word_count_vector.toarray(), columns=cv.get_feature_names_out())

    tfidf_transformer = TfidfTransformer()
    X = tfidf_transformer.fit_transform(word_count_vector)
    idf = pd.DataFrame({'feature_name':cv.get_feature_names_out(), 'idf_weights':tfidf_transformer.idf_})

    tf_idf = pd.DataFrame(X.toarray() ,columns=cv.get_feature_names_out())

    tf_idf_per_song = tf_idf.sum(axis=1)
    tf_idf_per_song.columns = ["tfidf"]
    return tf_idf_per_song
