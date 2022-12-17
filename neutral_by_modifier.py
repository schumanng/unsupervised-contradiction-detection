
"""
This script is used to create neutral hypothesis sentences by inserting neutral modifiers (adjectives).
A neutral modifier for the sentence "A man swims under a bridge" could be, for example, "elderly" (referring to the man)
and "wooden" (referring to the bridge). For this purpose, "MASK-tokens" are inserted at suitable places in the premise
sentence. The sentence is then passed to a Bert model, which attempts to fill these masked tokens with the help of its
"masked word prediction". The advantage of this approach is that Bert predicts the masked words based on the surrounding
words, which makes the inserted words fit well within the sentence (also in terms of tense and numerus).
The modified sentences are then transformed into alternative sentences with the same meaning using a paraphrasing
model. The resulting dataframe contains the original premise sentences in the first column and the newly generated
neutral hypothesis sentences in the columns 2 to n.

What you need to do: install required packages and change the path according to your directory (line 176)
"""

import spacy
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertTokenizer, BertForMaskedLM

torch_device = 'cpu'
pd.set_option('display.max_columns', 1000)
pd.set_option('display.max_colwidth', None)

# load spacy's language model "en_core_web_lg", Bert model for masked word prediction and pegasus model for paraphrasing
pegasus_model = AutoModelForSeq2SeqLM.from_pretrained("tuner007/pegasus_paraphrase")
pegasus_tokenizer = AutoTokenizer.from_pretrained("tuner007/pegasus_paraphrase")
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
model = BertForMaskedLM.from_pretrained('bert-base-cased')
nlp_model = spacy.load('en_core_web_lg')
model.eval()



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



# creates short sentences that focus on the newly added modifier
def short_sents_focusing_on_the_modifers (doc, masked_indices):
    modifier_index = 0
    sentence = None
    for index in masked_indices:
        if len(doc) >= index:
            modifier_within_sent = doc[index - 1]
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
        modifier_index += 1
    return sentence



# Places "[MASK]-tokens" at positions within the sentence where a neutral modifier (ADJ) makes sense.
def add_masked_labels (doc):
    new_sent = []
    for token in doc:
        # A modifier can typically be placed in front of a noun...
        if token.dep_ == "det" and token.pos_ == 'DET' and token.head.pos_ == 'NOUN':
            noun = token.head
            # ...provided the noun does not have another modifier yet.
            existing_modifier = [w for w in noun.lefts if w.pos_ == 'ADJ' and w.dep_ == "amod"]
            if not existing_modifier:
                new_sent.append(str(token))
                new_sent.append('[MASK]')
            else:
                new_sent.append(str(token))
        else:
            new_sent.append(str(token))
    new_sent = ' '.join(new_sent)
    new_sent = new_sent.replace(' ,', ',')
    if new_sent[-2:] == ' .':
        new_sent = new_sent[:-2] + '.'
    return new_sent



# Predicts masked words within a sentence based on the surrounding words (using BERT's masked word prediction).
def predict_masked_sent(text, nlp_model, top_k=5):
    # Tokenize input using spacy
    doc = nlp_model(text)
    text = add_masked_labels(doc)
    doc_as_list = list(nlp_model(text))
    text = "[CLS] %s [SEP]" % text
    tokenized_text = tokenizer.tokenize(text)
    masked_indices = [i for i, x in enumerate(tokenized_text) if x == "[MASK]"]
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    sentences = []

    new_modifiers = []
    for masked_index in masked_indices:
        # Predict all masked tokens
        with torch.no_grad():
            outputs = model(tokens_tensor)
            predictions = outputs[0]

        probs = torch.nn.functional.softmax(predictions[0, masked_index], dim=-1)
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

        for i, pred_idx in enumerate(top_k_indices):
            predicted_token = tokenizer.convert_ids_to_tokens([pred_idx])[0]
            new_modifiers.append(predicted_token)

    modifier_index = 0
    masked_indices = [i for i, x in enumerate(doc_as_list) if str(x) == "MASK"]
    for index in masked_indices:
        doc_as_list[index] = new_modifiers[modifier_index]
        modifier_index += 1

    doc_as_list = [str(i) for i in doc_as_list]
    new_sent = ' '.join(doc_as_list)
    new_sent = new_sent.replace(' ,', ',')
    if new_sent[-2:] == ' .':
        new_sent = new_sent[:-2] + '.'
    new_sent = new_sent.replace('[ ', '')
    new_sent = new_sent.replace(' ]', '')

    counter_8_to_6 = 0
    # get paraphrased versions of the new sentence...
    paraphrased = get_response(new_sent, 10, 10)
    for p_sent in paraphrased:
        contains_at_least_one_modifier = False
        for modifier in new_modifiers:
            modifier = ' ' + modifier + ' '
            if modifier in p_sent:
                contains_at_least_one_modifier = True
        # ...and add them if they meet the following conditions
        if contains_at_least_one_modifier:
            if len(p_sent) < len(new_sent):
                div = len(p_sent) / len(new_sent)
                if (0.8 > div >= 0.6) and counter_8_to_6 < 4:
                    sentences.append(p_sent)
                    counter_8_to_6 += 1
                elif div < 0.6:
                    sentences.append(p_sent)

    # also create short sentences that focus on the new modifiers
    doc_of_new_sent = nlp_model(new_sent)
    focused_on_the_modifiers = short_sents_focusing_on_the_modifers(doc_of_new_sent, masked_indices)
    sentences.append(focused_on_the_modifiers)

    if len(sentences) == 1 and sentences[0] is None:
        sentences = []

    return sentences




# load SNLI training set
train_df = pd.read_csv('input/snli_1.0_train.csv', sep=',')  # adjust the path according to your directory
train_df_sample = train_df#[:2000]  # draw a sample, if desired
train_df_sample = train_df_sample['sentence1']
train_df_sample = train_df_sample.drop_duplicates()
neutral_modifiers_inserted = []
changes_ct = 0
sent_ct = 0

print('start neutral-modifier-insertion using BERT')
for sent in train_df_sample:
    sent_level_lst = [sent]
    changed = False
    new_sent = predict_masked_sent(sent, nlp_model, top_k=1)
    if new_sent:
        for n_sent in new_sent:
            if n_sent != sent:
                sent_level_lst.append(n_sent)
                changed = True
    if changed:
        changes_ct += 1
    neutral_modifiers_inserted.append(sent_level_lst)
    if sent_ct % 50 == 0:
        print('sentences processed: ', sent_ct + 1, ' | newly created sentences: ',
              changes_ct, ' | example of original sentence: ',
              sent, '| and its new neutral sentences: ', new_sent)
    sent_ct += 1
print('newly created sentences: ',changes_ct, ' Total sentences:', len(train_df_sample))


# function that turns the neutral sentence-list into a dataframe
neutral_modifier_insertion_df = pd.DataFrame(neutral_modifiers_inserted)
counter = 0
for col in neutral_modifier_insertion_df.columns:
    neutral_modifier_insertion_df.rename(columns={col:'neutral_modifier_'+str(counter)},inplace=True)
    counter += 1
neutral_modifier_insertion_df = neutral_modifier_insertion_df.rename(columns={"neutral_modifier_0": "sent"})

# write final output to disc:
neutral_modifier_insertion_df.to_csv('output/neutral_by_modifier.csv', sep=';')
