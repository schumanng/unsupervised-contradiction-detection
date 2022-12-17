
"""
This script is used to create contradictory hypothesis sentences by changing numerical values.
For this purpose, numerical values such as "3" or "two" are searched for within the premise sentence and replaced
by random values. The modified sentences are then transformed into alternative sentences with the same meaning
using a paraphrasing model. The resulting dataframe contains the original premise sentences in the first column
and the newly generated contradictory hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 193 and 235)
"""

import re
import random
import spacy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

torch_device = 'cpu'
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

# load spacy's language model "en_core_web_lg" and pegasus model for paraphrasing
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
nlp_model = spacy.load('en_core_web_lg')



# returns n paraphrased versions of a given sentence
def get_response(input_text, num_return_sequences, num_beams):
    batch = pegasus_tokenizer([input_text], truncation=True, padding='longest', max_length=60, return_tensors="pt").to(
        torch_device)
    translated = pegasus_model.generate(**batch, max_length=60, num_beams=num_beams,
                                        num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = pegasus_tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text



# helper function to filter spans
def filter_spans(spans):
    get_sort_key = lambda span: (span.end - span.start, -span.start)
    sorted_spans = sorted(spans, key=get_sort_key, reverse=True)
    result = []
    seen_tokens = set()
    for span in sorted_spans:
        # Check for end - 1 here because boundaries are inclusive
        if span.start not in seen_tokens and span.end - 1 not in seen_tokens:
            result.append(span)
        seen_tokens.update(range(span.start, span.end))
    result = sorted(result, key=lambda span: span.start)
    return result



# creates short sentences that focus on the new numeric value
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

    # simplify the main sentence statement
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



# counts the numeric values in the sentence
def numberCount(nlp_model, sent):
    doc = nlp_model(sent)
    nbr_positions = []
    for token in doc:
        if token.pos_ == 'NUM' and str(token).lower() != 'one':
            nbr_positions.append(token.i)
    return nbr_positions



# searches for numerical values in the sentence and exchanges them for deviating values.
def numberSubstitution (nlp_model, sent):
    small_numbers = ['two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten']
    doc = nlp_model(sent)
    sentences = []
    potential_nbr = None
    # get the positions of the numbers within the sentence
    number_positions_lst = numberCount(nlp_model, sent)
    if len(number_positions_lst) > 0:
        for adj_position in number_positions_lst:
            sentence = []
            for token in doc:
                if token.i == adj_position:
                    if str(token).lower() in small_numbers:
                        selected_nbr = False
                        while not selected_nbr:
                            # create random numbers between 2 and 7
                            number = int(random.uniform(2, 7))
                            potential_nbr = small_numbers[number - 1]
                            if potential_nbr != str(token).lower():
                                upperCase = (str(token) != str(token).lower())
                                if upperCase:
                                    potential_nbr = potential_nbr.title()
                                sentence.append(potential_nbr)
                                selected_nbr = True
                    else:
                        numeric_string = re.sub("[^0-9.]", "", str(token))
                        if numeric_string != '' and numeric_string != '0':
                            try:
                                num = int(numeric_string)
                            except ValueError:
                                print('exception! --> numeric_string: ', numeric_string)
                                break
                            range_start = num - (num * 0.6)
                            range_end = num * 1.6
                            selected_nbr = False
                            while not selected_nbr:
                                number = random.uniform(range_start, range_end)
                                if number != num:
                                    number = int(number)
                                    sentence.append(str(number))
                                    selected_nbr = True
                        else:
                            sentence.append(str(token))
                else:
                    sentence.append(str(token))
            sentence = ' '.join(sentence)
            sentence = sentence.replace(' ,', ',')
            if sentence[-2:] == ' .':
                sentence = sentence[:-2] + '.'
            if sentence != sent:
                sentences.append(sentence)
                sub_sent = extract_sub_sentences(nlp_model, sentence)
                if sub_sent:
                    if str(potential_nbr) in sub_sent[0]:
                        if len(sub_sent[0]) < len(sentence):
                            valid_sub_sent = sub_sent[0].replace(',,', ',')
                            sentences.append(valid_sub_sent)
                paraphrased = get_response(sentence, 5, 5)
                for p_sent in paraphrased:
                    if (str(potential_nbr) in p_sent) and p_sent != sentence:
                        sentences.append(p_sent)
    return sentences




# load SNLI training set
train_df = pd.read_csv('input/snli_1.0_train.csv', sep = ',')
train_df_sample = train_df#[:2000]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()
number_subst_lst = []
changes_ct = 0
sent_ct = 0

print('Start number substitution')
for sent in train_df_sample:
    sent_level_lst = [sent]
    changed = False
    new_sent = numberSubstitution(nlp_model, sent)
    if new_sent:
        for n_sent in new_sent:
            if n_sent != sent:
                sent_level_lst.append(n_sent)
                changed = True
    if changed:
        changes_ct += 1
    number_subst_lst.append(sent_level_lst)
    if sent_ct % 50 == 0:
        print('sentences processed: ', sent_ct + 1, ' | newly created sentences: ',
              changes_ct, ' | example of original sentence: ',
              sent, '| and its new neutral sentences: ', new_sent)
    sent_ct += 1
print('newly created sentences: ',changes_ct, ' Total sentences:', len(train_df_sample))


# function that turns the contradicting sentence-list into a dataframe
number_subst_df = pd.DataFrame(number_subst_lst)
counter = 0
for col in number_subst_df.columns:
    number_subst_df.rename(columns={col:'number_subst_'+str(counter)},inplace=True)
    counter += 1
number_subst_df = number_subst_df.rename(columns={"number_subst_0": "sent"})
columns = len(number_subst_df.columns)
while columns <= 6:
    number_subst_df["number_subst_"+str(columns)] = None
    columns += 1

# write final output to disc:
number_subst_df.to_csv('output/contradiction_by_number_substitution.csv', sep=';')
