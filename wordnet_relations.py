#!/usr/bin/env python
# wordnet_relations.py
# Author: Julie Kallini

from nltk.corpus import wordnet as wn
from upos import upos


def get_wordnet_tag(upos_tag):
    """
    Return the equivalent wordnet POS tag for the given universal
    POS tag.

    @param upos_tag (str): universal POS tag
    @return (str): wordnet tag
    """
    if upos_tag == upos.ADJ:
        return wn.ADJ
    elif upos_tag == upos.VERB:
        return wn.VERB
    elif upos_tag == upos.NOUN:
        return wn.NOUN
    elif upos_tag == upos.ADV:
        return wn.ADV
    else:
        raise ValueError(
            "POS tag {} not in closed class categories".format(upos_tag))


def get_synsets(word1, word2, tag):
    """
    Returns synsets of word1 and the synsets of word2.

    @param word1 (str): English word
    @param word2 (str): English word
    @param tag (str): NLTK tag of word1 and word2
    @return (tuple): 2-tuple of lists of synsets for word1 and word2
    """
    pos = get_wordnet_tag(tag)
    return wn.synsets(word1, pos=pos), wn.synsets(word2, pos=pos)


def in_wordnet(word, tag):
    """
    Returns whether the word with the given tag is present in WordNet.

    @param word (str): English word
    @param tag (str): UPOS tag of word
    @return (bool): is the word in WordNet
    """
    pos = get_wordnet_tag(tag)
    return len(wn.synsets(word, pos=pos)) != 0


def synonyms(word1, word2, tag):
    """
    Returns whether word1 and word2 are synonyms by testing all possible
    synsets of word1 and word2.

    @param word1 (str): English word
    @param word2 (str): English word
    @param tag (str): NLTK tag of word1 and word2
    @return (nullable bool):
        - True if word1 and word2 are synonyms, False otherwise
        - None if word1 or word2 is not in WordNet
    """
    if word1.lower() == word2.lower():
        return False

    syns1, syns2 = get_synsets(word1, word2, tag)
    if len(syns1) == 0 or len(syns2) == 0:
        return None

    return not set(syns1).isdisjoint(set(syns2))


def antonyms(word1, word2, tag):
    """
    Returns whether word1 is an antonym of word2 by testing all possible
    synsets of word1 and word2.

    @param word1 (str): English word
    @param word2 (str): English word
    @param tag (str): NLTK tag of word1 and word2
    @return (nullable bool):
        - True if word1 and word2 are antonyms, False otherwise
        - None if word1 or word2 is not in WordNet
    """
    syns1, syns2 = get_synsets(word1, word2, tag)
    if len(syns1) == 0 or len(syns2) == 0:
        return None

    # Test relation among all pairs of synsets
    synsets1 = set([l for s in syns1 for l in s.lemmas()])
    for ss in syns2:
        for l in ss.lemmas():
            antonyms = l.antonyms()
            if not synsets1.isdisjoint(antonyms):
                return True

    return False


def relates(word1, word2, rel, tag):
    """
    Returns whether word1 relates to word2 by testing all possible
    synsets of word1 and word2, using the given relation function.

    @param word1 (str): English word
    @param word2 (str): English word
    @param rel (function): relation function
    @param tag (str): NLTK tag of word1 and word2
    @return (nullable bool):
        - True if relation function holds, False otherwise
        - None if word1 or word2 is not in WordNet
    """
    syns1, syns2 = get_synsets(word1, word2, tag)
    if len(syns1) == 0 or len(syns2) == 0:
        return None

    # Test relation among all pairs of synsets
    synsets1 = set(syns1)
    for ss in syns2:
        relations = set([i for i in ss.closure(rel)])
        if not synsets1.isdisjoint(relations):
            return True

    return False


def is_hypernym(word1, word2, tag):
    """
    Returns whether word1 is a hypernym of word2 by testing all possible
    synsets of word1 and word2.

    @param word1 (str): English word
    @param word2 (str): English word
    @param tag (str): NLTK tag of word1 and word2
    @return (nullable bool):
        - True if word1 is a hypernym of word2, False otherwise
        - None if word1 or word2 is not in WordNet
    """
    return relates(word1, word2, lambda s: s.hypernyms(), tag)


def get_co_hyponyms(synset):
    """
    Returns co-hyponyms of the given synset.

    @param synset (Synset): wordnet synset
    @return (set): co-hyponyms of the synset
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

    @param word1 (str): English word
    @param word2 (str): English word
    @param tag (str): NLTK tag of word1 and word2
    @return (nullable bool):
        - True if word1 and word2 are co-hyponyms, False otherwise
        - None if word1 or word2 is not in WordNet
    """
    if word1.lower() == word2.lower():
        return False

    syns1, syns2 = get_synsets(word1, word2, tag)
    if len(syns1) == 0 or len(syns2) == 0:
        return None

    # Test relation among all pairs of synsets
    synsets1 = set(syns1)
    for ss in syns2:
        co_hyponyms = get_co_hyponyms(ss)
        if not synsets1.isdisjoint(co_hyponyms):
            return True

    return False


def entails(word1, word2, tag):
    """
    Returns whether word1 entails word2 by testing all possible
    synsets of word1 and word2.

    @param word1 (str): English word
    @param word2 (str): English word
    @param tag (str): NLTK tag of word1 and word2
    @return (nullable bool):
        - True if word1 entails word2, False otherwise
        - None if word1 or word2 is not in WordNet
    """
    return relates(word2, word1, lambda s: s.entailments(), tag)
