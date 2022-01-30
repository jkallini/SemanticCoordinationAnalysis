#!/usr/bin/env python
# utils.py
# Author: Julie Kallini

import pandas as pd
from multipledispatch import dispatch
import spacy
import wordnet_relations as wr

# Conjunctions under analysis
CONJUNCTIONS = ['and', 'or', 'but', 'nor']

# Categories under analysis
NOUN_CATEGORIES = ['NN', 'NNS', 'NNP', 'NNPS', 'NP', 'NX']
VERB_CATEGORIES = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'VP']
ADJ_CATEGORIES = ['JJ', 'JJR', 'JJS', 'ADJP']
ADV_CATEGORIES = ['RB', 'RBR', 'RBS', 'ADVP']

PHRASAL_CATEGORIES = ['NP', 'VP', 'ADJP', 'ADVP']


@dispatch(str, str)
def pretty_print(input_file, output_file):
    '''
    Convert a CSV file of coordination phrases into a pretty-printed RTF file.

    @param input_file (str): path to the input CSV file.
    @param output_file (str): path to the output RTF file.
    '''

    input_df = pd.read_csv(input_file, index_col=None, header=0)
    pretty_print(input_df, output_file)


@dispatch(pd.DataFrame, str)
def pretty_print(input_df, output_file):
    '''
    Convert a DataFrame of coordination phrases into a pretty-printed RTF file.

    @param input_df (DataFrame): input coordination phrases
    @param output_file (str): path to the output RTF file
    '''

    out_file = open(output_file, 'w')
    out_file.write("{\\rtf1\n")

    section = ''
    for index, row in input_df.iterrows():
        sent = str(row['Sentence Text'])
        conj1 = str(row['1st Conjunct Text'])
        cat1 = str(row['1st Conjunct Category'])
        conj2 = str(row['2nd Conjunct Text'])
        cat2 = str(row['2nd Conjunct Category'])
        conjunction = str(row['Conjunction'])

        label = cat1 + "+" + cat2

        if index == 0:
            out_file.write('\\b ' + label + '\\b0\line\line\n')
            section = label
        elif section != label:
            out_file.write('\line\line\\b ' + label + '\\b0\line\line\n')
            section = label

        conj1_labeled = '\\b ' + '[' + cat1 + ' ' + conj1 + ']' + '\\b0 '
        conj2_labeled = '\\b ' + '[' + cat2 + ' ' + conj2 + ']' + '\\b0 '

        ccp = conj1 + ' ' + conjunction + ' ' + conj2
        ccp_labeled = conj1_labeled + ' ' + conjunction + ' ' + conj2_labeled
        sent = sent.replace(ccp, ccp_labeled)

        out_file.write(str(index + 1) + '. ' + sent + '\line\n')

    out_file.write("}")
    out_file.close()


def likes_by_category(df):
    '''
    Split a DataFrame of coordinations into four DataFrames containing
    like coordinations for each of the four open-class syntactic categories:
    nouns, verbs, adjectives, and adverbs.

    @param df (DataFrame): coordination phrases
    @return (4-tuple of DataFrames): (nouns, verbs, adjps, advps)
    '''

    nouns = df[(df['1st Conjunct Category'].isin(NOUN_CATEGORIES)) & (
        df['2nd Conjunct Category'].isin(NOUN_CATEGORIES))]

    verbs = df[(df['1st Conjunct Category'].isin(VERB_CATEGORIES)) & (
        df['2nd Conjunct Category'].isin(VERB_CATEGORIES))]

    adjps = df[(df['1st Conjunct Category'].isin(ADJ_CATEGORIES)) & (
        df['2nd Conjunct Category'].isin(ADJ_CATEGORIES))]

    advps = df[(df['1st Conjunct Category'].isin(ADV_CATEGORIES)) & (
        df['2nd Conjunct Category'].isin(ADV_CATEGORIES))]

    return (nouns, verbs, adjps, advps)


def likes_df(df):
    '''
    Returns a DataFrame of the like coordinations contained in the
    given DataFrame. Like coordinations must have conjuncts that are
    members of one of the four open-class syntactic categories:
    nouns, verbs, adjectives, and adverbs.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): like coordination phrases
    '''

    likes = pd.concat(list(likes_by_category(df)),
                      axis=0, ignore_index=True)

    return likes


def unlikes_df(df):
    '''
    Returns a DataFrame of the unlike coordinations contained in the
    given DataFrame.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): unlike coordination phrases
    '''
    
    df = df[df['1st Conjunct Category'].isin(PHRASAL_CATEGORIES)]
    df = df[df['2nd Conjunct Category'].isin(PHRASAL_CATEGORIES)]

    # Get unlike category combinations
    unlikes = df.loc[df['1st Conjunct Category']
                     != df['2nd Conjunct Category']]

    return unlikes


def filter_conj_length(df, length):
    '''
    Returns a DataFrame of coordinations contained in the given
    DataFrame where each conjunct is at most the given length.

    @param df (DataFrame): coordination phrases
    @param length (int): length threshold for filtering coordinations
    @return (DataFrame): filtered coordination phrases
    '''
    
    df = df.copy()

    df['Sentence Text'] = df['Sentence Text'].astype('str')
    mask1 = df['1st Conjunct Text'].str.split().str.len()
    mask2 = df['2nd Conjunct Text'].str.split().str.len()
    return df.loc[(mask1 <= length) & (mask2 <= length)]


def get_head(phrase, nlp):
    '''
    Returns the syntactic head of the phrase using spaCy's dependency
    parser, if it exists. Returns None otherwise.

    @param phrase (str): phrase to parse
    @param nlp (Doc): spaCy language model
    @return (str): head word of phrase
    '''

    doc = nlp(phrase)
    sents = list(doc.sents)
    if sents != []:
        return str(list(doc.sents)[0].root)


def add_conj_heads(df):
    '''
    Adds two new columns to the given DataFrame containing the syntactic
    heads of each conjunct.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): coordination phrases containing conjunct heads
    '''

    df = df.copy()

    nlp = spacy.load("en_core_web_lg")
    df['1st Conjunct Head'] = df.apply(
        lambda row: get_head(str(row['1st Conjunct Text']), nlp), axis=1)
    df['2nd Conjunct Head'] = df.apply(
        lambda row: get_head(str(row['2nd Conjunct Text']), nlp), axis=1)
    
    return df


def analyze_synonymy(df):
    """
    Run synonymy analysis on all categories in the given DataFrame.
    Returns DataFrame with new boolean 'Synonyms?' column.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): coordinations with synonymy relation column
    """

    df = df.copy()

    # Ensure Dataframe only contains conjuncts of like categories
    df = likes_df(df)

    df['Synonyms?'] = df.apply(lambda row: wr.synonyms(
        str(row['1st Conjunct Head']),
        str(row['2nd Conjunct Head']),
        str(row['1st Conjunct Category'])), axis=1)

    return df


def analyze_antonymy(df):
    """
    Run antonymy analysis on adjective and adverb-like categories
    in the given DataFrame. Returns DataFrame with a new 'Antonyms?'
    column.

    @param df (DataFrame): coordination phrases
    @return (DataFrame): coordinations with antonymy relation column
    """

    df = df.copy()

    # Ensure DataFrame only contains adjective/adverb categories
    df = df[(df['1st Conjunct Category'].isin(ADJ_CATEGORIES) & df['2nd Conjunct Category'].isin(ADJ_CATEGORIES)) |
            (df['1st Conjunct Category'].isin(ADV_CATEGORIES) & df['2nd Conjunct Category'].isin(ADV_CATEGORIES))]

    df['Antonyms?'] = df.apply(lambda row: wr.antonyms(
        str(row['1st Conjunct Head']),
        str(row['2nd Conjunct Head']),
        str(row['1st Conjunct Category'])), axis=1)

    return df


def analyze_hypernymy(df):
    """
    Run hypernymy analysis on noun-like and verb-like categories in the given
    DataFrame. Returns DataFrame with new columns '1st Conjunct Hypernym?'
    and '2nd Conjunct Hypernym?'.

    @param df (DataFrame): coordination phrases
    @return (Dataframe): coordinations with hypernymy relation columns
    """

    df = df.copy()

    # Ensure DataFrame only contains nominal/verbal categories
    df = df[(df['1st Conjunct Category'].isin(NOUN_CATEGORIES) & df['2nd Conjunct Category'].isin(NOUN_CATEGORIES)) |
            (df['1st Conjunct Category'].isin(VERB_CATEGORIES) & df['2nd Conjunct Category'].isin(VERB_CATEGORIES))]

    df['1st Conjunct Hypernym?'] = df.apply(lambda row: wr.is_hypernym(
        str(row['1st Conjunct Head']),
        str(row['2nd Conjunct Head']),
        str(row['1st Conjunct Category'])), axis=1)
    df['2nd Conjunct Hypernym?'] = df.apply(lambda row: wr.is_hypernym(
        str(row['2nd Conjunct Head']),
        str(row['1st Conjunct Head']),
        str(row['1st Conjunct Category'])), axis=1)

    return df


def analyze_cohyponymy(df):
    """
    Run co-hyponymy analysis on noun-like and verb-like categories in the given
    DataFrame. Returns DataFrame with new column 'Co-hyponyms?'.

    @param df (DataFrame): coordination phrases
    @return (Dataframe): coordinations with co-hyponymy relation column
    """

    df = df.copy()

    # Ensure DataFrame only contains nominal/verbal categories
    df = df[(df['1st Conjunct Category'].isin(NOUN_CATEGORIES) & df['2nd Conjunct Category'].isin(NOUN_CATEGORIES)) |
            (df['1st Conjunct Category'].isin(VERB_CATEGORIES) & df['2nd Conjunct Category'].isin(VERB_CATEGORIES))]

    df['Co-hyponyms?'] = df.apply(lambda row: wr.co_hyponyms(
        str(row['2nd Conjunct Head']),
        str(row['1st Conjunct Head']),
        str(row['1st Conjunct Category'])), axis=1)

    return df


def analyze_entailment(df):
    """
    Run entailment analysis on verb-like categories in the given DataFrame.
    Returns DataFrame with new columns '1st Conjunct Entails 2nd?' and
    '2nd Conjunct Entails 1st?'.

    @param df (DataFrame): coordination phrases
    @return (Dataframe): coordinations with entailment relation columns
    """

    df = df.copy()

    # Ensure conjuncts have verbal categories
    df = df[df['1st Conjunct Category'].isin(VERB_CATEGORIES)]
    df = df[df['2nd Conjunct Category'].isin(VERB_CATEGORIES)]

    df['1st Conjunct Entails 2nd?'] = df.apply(lambda row: wr.entails(
        str(row['1st Conjunct Head']),
        str(row['2nd Conjunct Head']),
        str(row['1st Conjunct Category'])), axis=1)
    df['2nd Conjunct Entails 1st?'] = df.apply(lambda row: wr.entails(
        str(row['2nd Conjunct Head']),
        str(row['1st Conjunct Head']),
        str(row['1st Conjunct Category'])), axis=1)

    return df


def _make_gen(reader):
    '''
    Helper generator function for rawgencount.
    '''
    b = reader(1024 * 1024)
    while b:
        yield b
        b = reader(1024*1024)


def rawgencount(filename):
    """
    Programmatically counts the number of lines in a file.

    @param filename (str): path to file
    @return (int): number of lines in file
    """
    f = open(filename, 'rb')
    f_gen = _make_gen(f.raw.read)
    return sum(buf.count(b'\n') for buf in f_gen)
