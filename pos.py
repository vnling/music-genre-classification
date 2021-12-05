import nltk
from nltk.tokenize import word_tokenize
from collections import  defaultdict
import numpy as np

def get_pos(lyrics):
    pos = []
    for lyric in lyrics:
        words = word_tokenize(lyric)
        pos_tags = nltk.pos_tag(words)

        pos_count = defaultdict(int)
        for entry in pos_tags:
            word, tag = entry
            pos_count[tag] += 1
        pos.append(pos_count)

    return np.array(pos)