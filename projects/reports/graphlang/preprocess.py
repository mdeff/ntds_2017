import numpy as np, nltk
from nltk.corpus import wordnet
from nltk.stem.snowball import SnowballStemmer
import string


wnl = nltk.WordNetLemmatizer()

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def words_tokens(text, ignore_punct=False):
    exclude = set(string.punctuation)
    return [t for t in nltk.word_tokenize(text) if not ignore_punct or t[0] not in set(string.punctuation)]

def words_lems(text, lower=False, ignore_punct=False):
    text_pos = nltk.pos_tag(words_tokens(text, ignore_punct=ignore_punct))
    text_lems = [wnl.lemmatize(t,pos=get_wordnet_pos(p)) for t,p in text_pos]

    if lower:
        return [lem.lower() for lem in text_lems]
    else:
        return text_lems

def stem_word(word, stemmer, lower=False):
    """
    Tries to stem a word using the provided stemmer. If an error occurs (happens sometimes with the arabic stemmer), return the word as is
    """
    try:
        if lower:
            return stemmer.stem(word).lower()
        else:
            return stemmer.stem(word)
    except Exception:
        if lower:
            return word.lower()
        else:
            return word

def words_stems(text, lang="english", lower=False, ignore_stopwords=False, ignore_punct=False):
    tokens = words_tokens(text, ignore_punct=ignore_punct)
    stemmer = SnowballStemmer(lang, ignore_stopwords=ignore_stopwords)

    return [stem_word(t, stemmer, lower) for t in tokens]

def words_to_int(words, first_index=0, ignore_punct=False, ignore_stopwords=False, lang=None):
    """
    Returns a dictionary mapping each of the words in the corpus to an integer
    """
    if(ignore_stopwords and lang != None):
        stopwords = nltk.corpus.stopwords.words(lang)
        words = filter(lambda w: w not in stopwords, words)

    if(ignore_punct):
        words_set = set([w for w in words if w not in string.punctuation])
    else:
        words_set = set(words)

    return {w:i for w,i in zip(words_set, range(first_index, first_index + len(words_set)))}
