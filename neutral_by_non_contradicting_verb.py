
"""
This script is used to create neutral hypothesis sentences by finding sentence pairs that have the same subject, verb
and object, but also additional adjectives or nouns. The resulting dataframe contains the original premise sentences
in the first column and the newly generated neutral hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 220)
"""

import re
import spacy
import numpy as np
import pandas as pd

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)

# load spacy's language model "en_core_web_lg"
nlp_model = spacy.load('en_core_web_lg')



# helper function to filter spans
def filter_spans(spans):
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result



# get subject and verb of a sentence
def sentence_sb_and_verb (nlp_model, sent):
    doc = nlp_model(sent)
    nouns = []
    for token in doc:
        if (token.pos_ == 'NOUN' and (token.dep_ == 'dobj' or token.dep_ == 'pobj')):
            noun_str = token.lemma_.lower()
            nouns.append(noun_str)

    sentence = [sent, str(nouns)]
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    no_subject = True
    for token in doc:
        if token.dep_ == "nsubj":
            if str(token) in ['the']:
                if token.head.pos_ == 'VERB' and token.head.dep_ == "rc":
                    verb = token.head
                    if verb.head.pos_ == 'NOUN':
                        if verb.head.dep_ == "ROOT":
                            sb = str(verb.head).lower()
                            cc_right = [w for w in verb.head.rights if w.dep_ == "cc" and w.pos_ == 'CCONJ']
                            if cc_right:
                                conj_right = [w for w in verb.head.rights if w.dep_ == "conj" and w.pos_ == 'NOUN']
                                if conj_right:
                                    sb = sb + ' ' + str(cc_right[0]).lower() + ' ' + str(conj_right[0]).lower()
                            sentence.append(sb)
                            no_subject = False
                            break

                        else:
                            left = verb.head
                            if left.dep_ == "cj" and left.head.pos_ == 'CCONJ':
                                cj_left = left.head
                                if cj_left.dep_ == "cd" and cj_left.head.pos_ == 'NOUN':
                                    sentence.append(str(cj_left.head) + ' ' + str(cj_left) + ' ' + str(verb.head))
                                    no_subject = False
                                    break

            else:
                sb = str(token).lower()
                cc_right = [w for w in token.rights if w.dep_ == "cc" and w.pos_ == 'CCONJ']
                if cc_right:
                    conj_right = [w for w in token.rights if w.dep_ == "conj" and w.pos_ == 'NOUN']
                    if conj_right:
                        sb = sb + ' ' + str(cc_right[0]).lower() + ' ' + str(conj_right[0]).lower()
                elif not cc_right:
                    conj_right = [w for w in token.rights if w.dep_ == "conj" and w.pos_ == 'NOUN']
                    if conj_right:
                        cc_right = [w for w in conj_right[0].rights if w.dep_ == "cc" and w.pos_ == 'CCONJ']
                        if cc_right:
                            conj_right_2 = [w for w in conj_right[0].rights if w.dep_ == "conj" and w.pos_ == 'NOUN']
                            if conj_right_2:
                                sb = sb + ' ' + str(conj_right[0]).lower() + ' ' + str(cc_right[0]).lower() + ' ' \
                                     + str(conj_right_2[0]).lower()
                    prep_right = [w for w in token.rights if w.dep_ == "prep" and w.pos_ == 'ADP']
                    if prep_right:
                        pobj_right = [w for w in prep_right[0].rights if w.dep_ == "pobj" and w.pos_ == 'NOUN']
                        if pobj_right:
                            sb = sb + ' ' + str(prep_right[0]).lower() + ' ' + str(pobj_right[0]).lower()

                sentence.append(sb)
                no_subject = False
                break

    if no_subject:
        sentence.append('-')
    no_verb = True
    for token in doc:
        if token.pos_ == "VERB":
            sentence.append(str(token.lemma_))
            no_verb = False
            break

    if no_verb:
        no_adverb = True
        for token in doc:
            if token.pos_ == "ADV":
                sentence.append(str(token.lemma_))
                no_adverb = False
                break
        if no_adverb:
            sentence.append('-')
    return sentence



# collect subject and verb for each sentence
def create_sentence_DB (nlp_model, sents):
    print('create_sentence_DB')
    sentence_DB = []
    not_valid_ct = 0
    idx = 0
    for sent in sents:
        if idx % 100 == 0:
            print('processed sentences: ', idx)
        sv_triple = sentence_sb_and_verb(nlp_model, sent)
        if sv_triple[1] != '-' and sv_triple[2] != '-':
            sentence_DB.append(sv_triple)
        else:
            not_valid_ct += 1
        idx += 1
    return sentence_DB



def find_sent_with_same_sb (sents_DB_df, own_sents_df, nlp_model):
    print('======================')
    print('find_sent_with_same_sb')
    valid_pairs = []
    for index, row in own_sents_df.iterrows():
        sents_added = 0
        if index % 100 == 0:
            print('processed sentences: ', index)
        sample_from_db = sents_DB_df
        row_lst = [[row['sent'], row['n_chunks'], row['subject'], row['verb']]]
        row_df = pd.DataFrame(row_lst, columns=['sent', 'n_chunks', 'subject', 'verb'])
        join_df = row_df.merge(sample_from_db, on=['subject', 'verb'], how='left')
        join_df = join_df.drop(join_df[join_df['sent_x'] == join_df['sent_y']].index)
        for index_inner, row_inner in join_df.iterrows():
            if sents_added < 5:
                if str(row_inner['n_chunks_x']) != 'nan' and str(row_inner['n_chunks_y']) != 'nan':
                    n_chunks_x_lst = row_inner['n_chunks_x']
                    n_chunks_x_lst = n_chunks_x_lst.replace('[', '')
                    n_chunks_x_lst = n_chunks_x_lst.replace(']', '')
                    n_chunks_x_lst = n_chunks_x_lst.replace("'", '')
                    n_chunks_x_lst = n_chunks_x_lst.split(', ')
                    n_chunks_y_lst = row_inner['n_chunks_y']
                    n_chunks_y_lst = n_chunks_y_lst.replace('[', '')
                    n_chunks_y_lst = n_chunks_y_lst.replace(']', '')
                    n_chunks_y_lst = n_chunks_y_lst.replace("'", '')
                    n_chunks_y_lst = n_chunks_y_lst.split(', ')
                    intersection = list(set(n_chunks_x_lst) & set(n_chunks_y_lst))
                    if n_chunks_x_lst == n_chunks_y_lst:
                        if len(intersection) == len(n_chunks_x_lst):
                            p1_sentence = row_inner['sent_x']
                            p2_sentence = row_inner['sent_y']
                            verbs_sent_1 = []
                            verbs_sent_2 = []
                            doc = nlp_model(p1_sentence)
                            for token in doc:
                                if token.pos_ == 'VERB':
                                    verbs_sent_1.append(str(token.lemma_))
                            doc = nlp_model(p2_sentence)
                            for token in doc:
                                if token.pos_ == 'VERB':
                                    verbs_sent_2.append(str(token.lemma_))
                            if verbs_sent_1 != verbs_sent_2:
                                sent_1_intersection = np.setdiff1d(verbs_sent_1, verbs_sent_2)
                                sent_2_intersection = np.setdiff1d(verbs_sent_2, verbs_sent_1)
                                row_to_append = row_inner

                                if (len(sent_2_intersection) > 0) and (len(sent_1_intersection) == 0):
                                    row_to_append = [p1_sentence, row_inner['n_chunks_x'][0], row_inner['subject'],
                                                     row_inner['verb'], p2_sentence, row_inner['n_chunks_y'][0],
                                                     row_inner['verb']]
                                elif (len(sent_1_intersection) > 0) and (len(sent_2_intersection) == 0):
                                    row_to_append = [p2_sentence, row_inner['n_chunks_x'][0], row_inner['subject'],
                                                     row_inner['verb'], p1_sentence, row_inner['n_chunks_y'][0],
                                                     row_inner['verb']]

                                valid_pairs.append(row_to_append)
                                sents_added += 1

    valid_df = pd.DataFrame(valid_pairs, columns=['sent_x',
                                                  'n_chunks_x',
                                                  'subject',
                                                  'verb_x',
                                                  'sent_y',
                                                  'n_chunks_y',
                                                  'verb_y'])
    return valid_df



# load SNLI training set
train_df = pd.read_csv('snli_1.0_train.csv', sep=',')  # adjust the path according to your directory
train_df_sample = train_df#[:40000]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()

# create two data sets of sentences
own_sents = create_sentence_DB(nlp_model, train_df_sample)
sents_DB = own_sents
sents_DB_df = pd.DataFrame(sents_DB, columns=['sent', 'n_chunks', 'subject', 'verb'])
own_sents_df = pd.DataFrame(own_sents, columns=['sent', 'n_chunks', 'subject', 'verb'])

# exclude sentences whose subject is a personal pronoun
exclude_lst = ['who', 'which', 'he', 'she', 'it', 'they']
for item in exclude_lst:
    sents_DB_df = sents_DB_df[sents_DB_df.subject != item]
sents_DB_df = sents_DB_df[sents_DB_df.verb != 'wear']
sents_DB_df = sents_DB_df[sents_DB_df.n_chunks != '[]']
for item in exclude_lst:
    own_sents_df = own_sents_df[own_sents_df.subject != item]
own_sents_df = own_sents_df[own_sents_df.verb != 'wear']
own_sents_df = own_sents_df[own_sents_df.n_chunks != '[]']

# find sentence with the same subject, and at least one overlapping verb and noun chunk
join_df = find_sent_with_same_sb(sents_DB_df, own_sents_df, nlp_model)
join_df = join_df.sort_values('sent_x')

# reduce to the first 5
join_df_top_5 = join_df.groupby('sent_x').head(5)
join_df_top_5 = join_df_top_5.reset_index()
join_df_top_5 = join_df_top_5.assign(col_name='')
col_idx = 1
sent_ct = 0

# create final dataframe
print('================')
for (index, row), ii in zip(join_df_top_5.iterrows(), range(len(join_df_top_5.index))):
    if ii < len(join_df_top_5.index)-1:
        if join_df_top_5.iloc[ii+1]['sent_x'] != join_df_top_5.iloc[ii]['sent_x']:
            join_df_top_5.at[ii, 'col_name'] = 'non_contradicting_verb_' + str(col_idx)
            col_idx = 1
        else:
            join_df_top_5.at[ii, 'col_name'] = 'non_contradicting_verb_' + str(col_idx)
            col_idx += 1
    if sent_ct % 50 == 0:
        print('processed sentences:', sent_ct)
    sent_ct += 1

join_df_top_5.drop(join_df_top_5.tail(1).index,inplace=True) # drop last row
# take care of the last row-value (could not be handled in the loop due to index-out-of-bound)
last_row_idx = len(join_df_top_5.index)-1
if join_df_top_5.loc[last_row_idx,'sent_x'] == join_df_top_5.loc[last_row_idx-1,'sent_x']:
    col_number = re.sub("[^0-9]", "", str(join_df_top_5.loc[last_row_idx-1,'sent_x']))
    join_df_top_5.loc[last_row_idx, 'sent_x'] = 'non_contradicting_verb_' + str(col_number)
elif join_df_top_5.loc[last_row_idx,'sent_x'] != join_df_top_5.loc[last_row_idx-1,'sent_x']:
    join_df_top_5.loc[last_row_idx, 'col_name'] = 'non_contradicting_verb_1'
# pivot the table
pivoted = join_df_top_5.pivot(index="sent_x", columns ="col_name", values="sent_y")
pivoted = pivoted.reset_index().rename_axis(None, axis=1)
non_contradicting_verb_df = pivoted.rename(columns={"sent_x": "sent"})

# write final output to disc:
non_contradicting_verb_df.to_csv('output/neutral_by_non_contradicting_verb.csv', sep=';')
