
"""
This script is used to create contradictory hypothesis sentences based on antonyms of adjectives and adverbs.
For this purpose, corresponding antonyms are searched for all adjectives and adverbs occurring in the premise
sentence (using Wordnet). If antonyms are found, they are exchanged with the original adjectives/adverbs.
The modified sentences are then transformed into alternative sentences with the same meaning using a paraphrasing
model. The resulting dataframe contains the original premise sentences in the first column and the newly generated
contradictory hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 179 and 223)
"""

import spacy
import pandas as pd
from nltk.corpus import wordnet as wn
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



# returns - if available - the antonym to a given word
def antonyms_for(word):
    antonyms = set()
    for ss in wn.synsets(word):
        for lemma in ss.lemmas():
            any_pos_antonyms = [antonym.name() for antonym in lemma.antonyms()]
            for antonym in any_pos_antonyms:
                antonym_synsets = wn.synsets(antonym)
                if wn.ADJ not in [ss.pos() for ss in antonym_synsets]:
                    continue
                antonyms.add(antonym)
    return antonyms



# searches for specified morph type in given list
def getMorphType(morph_list, morph_type):
    for item in morph_list:
        if morph_type in str(item):
            return item



# counts the adjectives and adverbs in the sentence
def AdjCount(nlp_model, sent):
    doc = nlp_model(sent)
    adj_positions = []
    for token in doc:
        if token.pos_ == 'ADJ' or token.pos_ == 'ADV':
            adj_positions.append(token.i)
    return adj_positions



# creates short sentences that focus on the newly added antonyms.
def short_sentence_focusing_on_the_antonym(doc, antonym_index_lst):
    antonym_index = antonym_index_lst[0]
    sentence = None
    if len(doc) >= antonym_index:
        modifier_within_sent = doc[antonym_index]
        if modifier_within_sent.dep_ == "amod":
            if modifier_within_sent.head.pos_ == 'NOUN':
                number = getMorphType(modifier_within_sent.head.morph, 'Number')
                is_are = 'is'
                if number == 'Number=Sing':
                    is_are = 'is'
                elif number == 'Number=Plur':
                    is_are = 'are'
                sentence = 'The' + ' ' + str(modifier_within_sent.head) + ' ' + is_are + ' ' + str(
                    modifier_within_sent) + '.'
    return sentence



# creates contradicting sentences to a given sentence in which the adjectives / adverbs have been replaced by their antonyms.
def antonym_adj_adv(nlp_model, sent):
    sentences = []
    doc = nlp_model(sent)

    # get the positions of the adjectives and adverbs within the sentence
    adj_positions_lst = AdjCount(nlp_model, sent)
    if len(adj_positions_lst) > 0:
        for adj_position in adj_positions_lst:
            sentence = []
            original_adj = None
            antonym = None
            for token in doc:
                if token.i == adj_position:
                    token_str = str(token)
                    original_adj = token_str
                    antonyms = antonyms_for(token.lemma_)
                    if antonyms:
                        antonym = []
                        for word in antonyms:
                            antonym.append(word)
                        antonym = antonym[0]
                        if list(token.morph):
                            degree = getMorphType(list(token.morph), 'Degree')
                            if degree:
                                if degree == 'Degree=Cmp':
                                    if token_str[-2:] == 'er' and antonym[-2:] != 'er':
                                        antonym = antonym + token_str[-2:]

                            if len(sentence) > 0:
                                token_starts_with_aoe = antonym[:1] in ['a', 'o', 'e', 'u']
                                if token_starts_with_aoe and sentence[-1] == 'A':
                                    sentence = sentence[:-1]
                                    sentence.append('An')
                                elif token_starts_with_aoe and sentence[-1] == 'a':
                                    sentence = sentence[:-1]
                                    sentence.append('an')
                                elif not token_starts_with_aoe and sentence[-1] == 'An':
                                    sentence = sentence[:-1]
                                    sentence.append('A')
                                elif not token_starts_with_aoe and sentence[-1] == 'an':
                                    sentence = sentence[:-1]
                                    sentence.append('a')
                                elif not token_starts_with_aoe and antonym in ['many', 'various']:
                                    sentence = sentence[:-1]
                            sentence.append(antonym)

                        else:
                            antonym = token_str
                            sentence.append(antonym)
                    else:
                        sentence.append(token_str)
                else:
                    sentence.append(str(token))

            sentence = ' '.join(sentence)
            sentence = sentence.replace(' ,', ',')
            if sentence[-2:] == ' .':
                sentence = sentence[:-2] + '.'
            if (sentence != sent) and antonym:
                sentences.append(sentence)

                # get paraphrased versions of the new sentence...
                paraphrased = get_response(sentence, 7, 7)
                for p_sent in paraphrased:
                    if len(p_sent) < len(sentence):
                        div = len(p_sent) / len(sentence)
                        # ...and add them if they meet the following conditions
                        if (str(antonym) in p_sent) and (original_adj not in p_sent) and (p_sent != sentence) and (
                                div < 0.8):
                            sentences.append(p_sent)

                # also get short sentences that focus on the newly added antonym
                doc_of_new_sent = nlp_model(sentence)
                index_of_antonym = [i for i, x in enumerate(doc_of_new_sent) if str(x) == antonym]
                if index_of_antonym:
                    focused_on_the_antonym = short_sentence_focusing_on_the_antonym(doc_of_new_sent, index_of_antonym)
                    sentences.append(focused_on_the_antonym)

    return sentences




# load SNLI training set
train_df = pd.read_csv('input/snli_1.0_train.csv', sep=',')  # adjust the path according to your directory
train_df_sample = train_df#[:2000]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()
antonym_adj_adv_lst = []
changes_ct = 0
sent_ct = 0

print('start text generation for antonyms of adjectives and adverbs')
for sent in train_df_sample:
    sent_level_lst = [sent]
    changed = False
    new_sent = antonym_adj_adv(nlp_model, sent)
    if new_sent:
        for n_sent in new_sent:
            if n_sent != sent:
                sent_level_lst.append(n_sent)
                changed = True
    if changed:
        changes_ct += 1
    antonym_adj_adv_lst.append(sent_level_lst)
    if sent_ct % 50 == 0:
        print('sentences processed: ', sent_ct+1, ' | newly created sentences: ',
              changes_ct, ' | example of original sentence: ',
              sent, '| and its new contradicting sentences: ', new_sent)
    sent_ct += 1
print('newly created sentences: ',changes_ct, ' Total sentences:', len(train_df_sample))


# turns the contradicting sentence-list into a pandas dataframe
antonym_adj_adv_df = pd.DataFrame(antonym_adj_adv_lst)
counter = 0
for col in antonym_adj_adv_df.columns:
    antonym_adj_adv_df.rename(columns={col: 'adj_antonym_' + str(counter)}, inplace=True)
    counter += 1
antonym_adj_adv_df = antonym_adj_adv_df.rename(columns={"adj_antonym_0": "sent"})
columns = len(antonym_adj_adv_df.columns)
while columns <= 7:
    antonym_adj_adv_df["adj_antonym_" + str(columns)] = None
    columns += 1
while len(antonym_adj_adv_df.columns) > 7:
    antonym_adj_adv_df = antonym_adj_adv_df.drop(antonym_adj_adv_df.columns[len(antonym_adj_adv_df.columns) - 1],axis=1)

# write final output to disc:
antonym_adj_adv_df.to_csv('output/contradiction_by_antonym_adj_adv.csv', sep=';')
