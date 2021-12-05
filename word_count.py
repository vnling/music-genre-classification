import pandas as pd

def get_word_count(lyrics):
    counts = [len(l) for l in lyrics]
    return pd.DataFrame(counts, columns=['word_count'])