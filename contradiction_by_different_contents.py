
"""
This script is used to create contradictory hypothesis sentences by searching for completely different topics.
For this purpose, all nouns and their hypernyms are collected for each sentence within the training data set.
Then, pairs of sentences are selected in which there is no overlap of nouns and hypernyms (i.e. they have completely
different nouns and hypernyms). This is to ensure that the sentences differ sufficiently thematically.
The resulting dataframe contains the original premise sentences in the first column and the contradictory
hypothesis sentences in the columns 2 to 4.

What you need to do: install required packages and change the path according to your directory (line 83)
"""

import spacy
import pandas as pd
from nltk.corpus import wordnet as wn

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

# load spacy's language model "en_core_web_lg"
nlp_model = spacy.load('en_core_web_lg')



# get all the nouns and their hypernyms for each sentence
def create_noun_db (sents):
    print()
    print('get all the nouns and their hypernyms for each sentence')
    main_list = []
    idx = 0
    for sent in sents:
        if idx % 500 == 0:
            print('processed sentences: ', idx)
        idx += 1
        sents_and_their_nouns = [sent]
        list_str = ''
        doc = nlp_model(sent)
        for token in doc:
            if token.pos_ == 'NOUN':
                token_str = str(token.lemma_).lower()
                list_str = list_str + ', ' + token_str
                term_synset = wn.synsets(token_str)
                if term_synset:
                    hypernym_list = term_synset[0].hypernyms()
                    for hypernym in hypernym_list:
                        hypernym_cleaned = str(hypernym)[8:str(hypernym).find('.')]
                        list_str = list_str + ', ' + hypernym_cleaned

        list_str = list_str[2:]
        sents_and_their_nouns.append(list_str)
        main_list.append(sents_and_their_nouns)
    return main_list



# finds sentence pairs that do not have any of the same nouns/hypernyms (three sentences at most)
def sents_with_different_nouns (sents_DB_1_df, sents_DB_2_df):
    print()
    print('find sentence pairs that do not have any of the same nouns/hypernyms (three sentences at most)')
    contradicting_sents = []
    idx = 0
    for index, row_df1 in sents_DB_1_df.iterrows():
        if idx % 500 == 0:
            print('processed sentences: ', idx)
        idx += 1
        sent_triple = [row_df1['sent']]
        counter = 0
        noun_list_DB1 = row_df1['nouns'].split(', ')
        sents_DB_2_df = sents_DB_2_df.sample(n=len(sents_DB_2_df), random_state=index)
        for index, row_df2 in sents_DB_2_df.iterrows():
            if counter == 3:
                break
            noun_list_DB2 = row_df2['nouns'].split(', ')
            if not any(noun in noun_list_DB2 for noun in noun_list_DB1):
                sent_triple.append(row_df2['sent'])
                counter += 1
        contradicting_sents.append(sent_triple)
    return contradicting_sents



# load SNLI training set
train_df = pd.read_csv('input/snli_1.0_train.csv', sep=',')  # adjust the path according to your directory
train_df_sample = train_df#[:2000]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()

print('start finding entirely different sentences')
print('==========================================')
sents_DB_1 = create_noun_db(train_df_sample)
sents_DB_1_df = pd.DataFrame(sents_DB_1, columns = ['sent', 'nouns'])
contradicting_sents = sents_with_different_nouns(sents_DB_1_df, sents_DB_1_df)
contradicting_sents_df = pd.DataFrame(contradicting_sents, columns = ['sent',
                                                                      'different_content_1',
                                                                      'different_content_2',
                                                                      'different_content_3'])
# write final output to disc:
contradicting_sents_df.to_csv('output/contradiction_by_different_contents.csv', sep=';')
