
"""
This script is used to create contradictory hypothesis sentences based on negation insertions.
For this purpose, a suitable place for a negation is searched for in the premise sentence and when
this is found, the appropriate negation form for the respective tense and numerus is inserted.
The modified sentences are then transformed into alternative sentences with the same meaning using a paraphrasing
model. The resulting dataframe contains the original premise sentences in the first column and the newly generated
contradictory hypothesis sentences in the columns 1 to n.

What you need to do: install required packages and change the path according to your directory (line 182 and 220)
"""

import spacy
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
torch_device = 'cpu'

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



# searches for specified morph type in given list
def getMorphType(morph_list, morph_type):
    for item in morph_list:
        if morph_type in str(item):
            return item



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



# creates contradicting sentences to a given sentence by inserting negations ("not")
def negationIntroduction (nlp_model, sent):
    doc = nlp_model(sent)
    spot_for_negation = []
    no_introduction = True

    # merge entities and noun chunks into one token
    spans = list(doc.ents) + list(doc.noun_chunks)
    spans = filter_spans(spans)
    with doc.retokenize() as retokenizer:
        for span in spans:
            retokenizer.merge(span)

    sentence = []

    # find positions in the sentence where inserting a negation makes sense (line 70 to 145):
    for token in doc:

        if token.pos_ == 'VERB' and token.dep_ == 'ROOT':
            aux_left = [w for w in token.lefts if w.dep_ == "aux"]
            nsubj_left = [w for w in token.lefts if w.dep_ == "nsubj"]

            if aux_left:
                if aux_left[0].pos_ == 'AUX':
                    sentence.append('not')
                    sentence.append(str(token))
                    no_introduction = False

            elif nsubj_left:
                if nsubj_left[0].pos_ == 'NOUN':
                    cc_right = [w for w in nsubj_left[0].rights if w.dep_ == "cc" and w.pos_ == 'CCONJ']
                    conj_right = [w for w in nsubj_left[0].rights if w.dep_ == "conj" and w.pos_ == 'NOUN']
                    number = getMorphType(list(nsubj_left[0].morph), 'Number')
                    if number == 'Number=Sing':
                        if cc_right or conj_right:
                            aspect = getMorphType(list(token.morph), 'Aspect')
                            if aspect == 'Aspect=Prog':
                                sentence.append('are not')
                            else:
                                sentence.append('do not')
                            sentence.append(str(token))
                            no_introduction = False

                        elif not cc_right:
                            aspect = getMorphType(list(token.morph), 'Aspect')
                            tense = getMorphType(list(token.morph), 'Tense')
                            verb = ''
                            if tense == 'Tense=Pres' and number == 'Number=Sing':
                                sentence.append('does not')
                                verb = str(token.lemma_)
                            elif tense == 'Tense=Pres' and number == 'Number=Plur':
                                sentence.append('do not')
                                verb = str(token)
                            elif tense == 'Tense=Pres' and aspect == 'Aspect=Prog':
                                sentence.append('not')
                                verb = str(token)
                            sentence.append(verb)
                            no_introduction = False

                    elif number == 'Number=Plur':
                        sentence.append('do not')
                        sentence.append(str(token))
                        no_introduction = False

        elif token.pos_ == 'VERB' and token.dep_ != 'ROOT':
            sentence.append(str(token))
        else:
            sentence.append(str(token))

    if no_introduction and spot_for_negation:
        sentence = []
        idx = 0
        for token in doc:
            sentence.append(str(token))
            if idx == spot_for_negation[0]:
                sentence.append('not')
            idx += 1

    if no_introduction and not spot_for_negation:
        sentence = []
        nbr_of_spots_for_negation = len([w for w in doc if (w.pos_ == 'ADP' and w.dep_ == "mo")])
        spots_visited = nbr_of_spots_for_negation - 1
        for token in doc:
            nk_right = [w for w in token.rights if (w.dep_ == "nk" and w.pos_ == "DET")]
            if token.pos_ == 'ADP' and token.dep_ == 'mo' and token.head.pos_ == 'VERB' and not nk_right:
                if nbr_of_spots_for_negation == 1:
                    sentence.append('not')
                if nbr_of_spots_for_negation > 1:
                    if spots_visited == nbr_of_spots_for_negation:
                        sentence.append('not')
                    spots_visited += 1
                sentence.append(str(token))
            else:
                sentence.append(str(token))

    sentence = ' '.join(sentence)
    sentence = sentence.replace(' ,', ',')

    if sentence[-2:] == ' .':
        sentence = sentence[:-2] + '.'
    if sent == sentence:
        sentence = []
    elif sent != sentence:
        sentence = [sentence]

        # get paraphrased versions of the new sentence...
        paraphrased = get_response(sentence[0], 5, 5)
        for p_sent in paraphrased:
            if len(p_sent) < len(sentence[0]):
                div = len(p_sent) / len(sentence[0])
                # ...and add them if they meet the following conditions
                if (' not ' in p_sent) and (p_sent != sentence[0]) and div <= 0.8:
                    sentence.append(p_sent)
    return sentence




# load SNLI training set
train_df = pd.read_csv('snli_1.0_train.csv', sep=',')  # adjust the path according to your directory
train_df_sample = train_df[:2000]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()
negation_insertion_lst = []
changes_ct = 0
sent_ct = 0

print('start text generation for inserting negations')
for sent in train_df_sample:
    sent_level_lst = [sent]
    changed = False
    new_sent = negationIntroduction(nlp_model, sent)
    if new_sent:
        for n_sent in new_sent:
            if n_sent != sent:
                sent_level_lst.append(n_sent)
                changed = True
    if changed:
        changes_ct += 1
    negation_insertion_lst.append(sent_level_lst)
    if sent_ct % 50 == 0:
        print('sentences processed: ', sent_ct + 1, ' | newly created sentences: ',
              changes_ct, ' | example of original sentence: ',
              sent, '| and its new contradicting sentences: ', new_sent)
    sent_ct += 1
print('newly created sentences: ',changes_ct, ' Total sentences:', len(train_df_sample))


# turns the contradicting sentence-list into a pandas dataframe
negation_insertion_df = pd.DataFrame(negation_insertion_lst)
counter = 0
for col in negation_insertion_df.columns:
    negation_insertion_df.rename(columns={col:'negation_'+str(counter)},inplace=True)
    counter += 1
negation_insertion_df = negation_insertion_df.rename(columns={"negation_0": "sent"})

# write final output to disc:
negation_insertion_df.to_csv('contradiction_by_negation.csv', sep=';')
