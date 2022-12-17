
"""
This script is used to create entailing hypothesis sentences by extracting sub-sentences that contain the essential
fact of the sentence. The extraction of the corresponding sentence subtrees is done using Spacy's dependency- and
part-of-speech tags.

What you need to do: install required packages and change the path according to your directory (line 95 and 133)
"""

import spacy
import pandas as pd

pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

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



# creates a short version of the original sentence that contains an essential fact of the sentence.
def extract_sub_sentences (nlp_model, sent):
    doc = nlp_model(sent)
    # Merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    new_sents = []
    sentence = []
    prep_subtrees = []

    for token in doc:
        prep_subtree = []
        head_contains_prep = False
        w = token
        while w.dep_ != 'ROOT':
            w = w.head
            if w.dep_ == "prep":
                prep_subtree.append(str(w))
                head_contains_prep = True
        if token.dep_ == "prep":
            for sub_t in token.subtree:
                prep_subtree.append(str(sub_t.text))
            prep_subtree = ' '.join(prep_subtree)
            prep_subtree = prep_subtree.replace(' ,', ',')
            if prep_subtree[-2:] == ' .':
                prep_subtree = prep_subtree[:-2] + '.'
            prep_subtree = prep_subtree.replace('[ ', '')
            prep_subtree = prep_subtree.replace(' ]', '')
            prep_subtrees.append(prep_subtree)

        if token.dep_ != "prep" and not head_contains_prep:
            sentence.append(str(token))

    new_sent = ' '.join(sentence)
    new_sent = new_sent.replace(' ,', ',')
    if new_sent[-2:] == ' .':
        new_sent = new_sent[:-2] + '.'
    new_sent = new_sent.replace('[ ', '')
    new_sent = new_sent.replace(' ]', '')

    # check whether the resulting sentence is still a valid sentence
    doc = nlp_model(new_sent)
    valid = False
    if new_sent != sent:
        for token in doc:
            if token.pos_ == 'AUX' or token.pos_ == 'VERB':
                valid = True
        if valid:
            new_sents.append(new_sent)

    return new_sents



# load SNLI training set
train_df = pd.read_csv('input/snli_1.0_train.csv', sep = ',')  # adjust the path according to your directory
train_df_sample = train_df#[:2000]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()
sub_sentences_lst = []
changes_ct = 0
sent_ct = 0

print('Start subsentence extraction')
for sent in train_df_sample:
    sent_level_lst = [sent]
    new_sent = extract_sub_sentences(nlp_model, sent)
    changed = False
    if new_sent:
        for n_sent in new_sent:
            if n_sent != sent:
                sent_level_lst.append(n_sent)
                changed = True
    if changed:
        changes_ct += 1
    sub_sentences_lst.append(sent_level_lst)
    if sent_ct % 50 == 0:
        print('sentences processed: ', sent_ct + 1, ' | newly created sentences: ',
              changes_ct, ' | example of original sentence: ',
              sent, '| and its new contradicting sentences: ', new_sent)
    sent_ct += 1
print('newly created sentences: ', changes_ct, ' Total sentences:', len(train_df_sample))


# function that turns subsentence-list into a dataframe
sub_sentences_df = pd.DataFrame(sub_sentences_lst)
counter = 0
for col in sub_sentences_df.columns:
    sub_sentences_df.rename(columns={col:'sub_sent_'+str(counter)},inplace=True)
    counter += 1
sub_sentences_df = sub_sentences_df.rename(columns={"sub_sent_0": "sent"})

# write final output to disc:
sub_sentences_df.to_csv('output/entailment_by_subsentence.csv', sep=';')
