#!/usr/bin/env python
# wordnet_relations.py

from nltk.corpus import wordnet as wn
import pandas as pd

NOUN_CATEGORIES = ['NN', 'NNS', 'NNP', 'NNPS', 'NP', 'NX']
VERB_CATEGORIES = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VP']
ADJ_CATEGORIES = ['JJ', 'JJR', 'JJS', 'ADJP']
ADV_CATEGORIES = ['RB', 'RBR', 'RBS', 'ADVP']


def get_wordnet_tag(nltk_tag):
    """
    Return the equivalent wordnet POS tag for the given nltk
    POS tag.

    Keyword Arguments:
      nltk_tag -- string NLTK tag
    Return:
      string wordnet tag
    """
    if nltk_tag in ADJ_CATEGORIES:
        return wn.ADJ
    elif nltk_tag in VERB_CATEGORIES:
        return wn.VERB
    elif nltk_tag in NOUN_CATEGORIES:
        return wn.NOUN
    elif nltk_tag in ADV_CATEGORIES:
        return wn.ADV
    else:
        # Use noun as a default POS tag in lemmatization
        return wn.NOUN


def synonyms(word1, word2, tag):
    """
    Returns whether word1 and word2 are synonyms by testing all possible
    synsets of word1 and word2.

    Keyword Arguments:
      word1 -- string
      word2 -- string
      tag -- string NLTK tag of word1 and word2
    Return:
      boolean
    """
    pos = get_wordnet_tag(tag)
    return not set(wn.synsets(word1, pos=pos)).isdisjoint(set(wn.synsets(word2, pos=pos)))


def antonyms(word1, word2, tag):
    """
    Returns whether word1 is an antonym of word2 by testing all possible
    synsets of word1 and word2.

    Keyword Arguments:
      word1 -- string
      word2 -- string
      tag -- string NLTK tag of word1 and word2
    Return:
      boolean
    """
    pos = get_wordnet_tag(tag)

    # Test relation among all pairs of synsets
    synsets1 = set([l for s in wn.synsets(word1, pos=pos) for l in s.lemmas()])
    for ss in wn.synsets(word2, pos=pos):
        for l in ss.lemmas():
            antonyms = l.antonyms()
            if not synsets1.isdisjoint(antonyms):
                return True

    return False


def relates(word1, word2, rel, tag):
    """
    Returns whether word1 relates to word2 by testing all possible
    synsets of word1 and word2, using the given relation function.

    Keyword Arguments:
      word1 -- string
      word2 -- string
      rel -- relation function
      tag -- string NLTK tag of word1 and word2
    Return:
      boolean
    """
    pos = get_wordnet_tag(tag)

    # Test relation among all pairs of synsets
    synsets1 = set(wn.synsets(word1, pos=pos))
    for ss in wn.synsets(word2, pos=pos):
        relations = set([i for i in ss.closure(rel)])
        if not synsets1.isdisjoint(relations):
            return True

    return False


def is_hypernym(word1, word2, tag):
    """
    Returns whether word1 is a hypernym of word2 by testing all possible
    synsets of word1 and word2.

    Keyword Arguments:
      word1 -- string
      word2 -- string
      tag -- string NLTK tag of word1 and word2
    Return:
      boolean
    """
    return relates(word1, word2, lambda s: s.hypernyms(), tag)


def get_co_hyponyms(synset):
    """
    Returns co-hyponyms of the given synset.

    Keyword Arguments:
      synset -- wordnet synset
    Return:
      set of co-hyponyms
    """
    co_hyponyms = set()
    for hyper in synset.hypernyms():
        for hypo in hyper.hyponyms():
            co_hyponyms.add(hypo)
    return co_hyponyms


def co_hyponyms(word1, word2, tag):
    """
    Returns whether word1 and word2 are co-hyponyms by testing all possible
    synsets of word1 and word2.

    Keyword Arguments:
      word1 -- string
      word2 -- string
      tag -- string NLTK tag of word1 and word2
    Return:
      boolean
    """
    pos = get_wordnet_tag(tag)

    # Test relation among all pairs of synsets
    synsets = set(wn.synsets(word1, pos=pos))
    for ss in wn.synsets(word2, pos=pos):
        co_hyponyms = get_co_hyponyms(ss)
        if not synsets.isdisjoint(co_hyponyms):
            return True

    return False


def entails(word1, word2, tag):
    """
    Returns whether word1 entails word2 by testing all possible
    synsets of word1 and word2.

    Keyword Arguments:
      word1 -- string
      word2 -- string
      tag -- string NLTK tag of word1 and word2
    Return:
      boolean
    """
    return relates(word2, word1, lambda s: s.entailments(), tag)