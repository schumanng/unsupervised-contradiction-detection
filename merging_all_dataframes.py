
"""
This script is used to transform the single data frames into a coherent training data set, which then has the
classic SNLI format: "sentence1" | "sentence2" | "gold_label".

What you need to do: change the path according to your directory (line 12, 251 and 403)
"""

import pandas as pd

# load all contradicting, entailing and neutral dataframes
path = 'input'  # adjust path
contradiction_by_antonym_adj_adv = pd.read_csv(path + '/contradiction_by_antonym_adj_adv.csv', sep=';')
contradiction_by_different_contents = pd.read_csv(path + '/contradiction_by_different_contents.csv', sep=';')
contradiction_by_different_verbs = pd.read_csv(path + '/contradiction_by_different_verbs.csv', sep=';')
contradiction_by_negation = pd.read_csv(path + '/contradiction_by_negation.csv', sep=';')
neutral_by_modifier = pd.read_csv(path + '/neutral_by_modifier.csv', sep=';')
contradiction_by_number_substitution = pd.read_csv(path + '/contradiction_by_number_substitution.csv', sep=';')
contradiction_by_different_object = pd.read_csv(path + '/contradiction_by_different_object.csv', sep=';')
entailment_by_paraphrasing = pd.read_csv(path + '/entailment_by_paraphrasing.csv', sep=';')
entailment_by_subsentence = pd.read_csv(path + '/entailment_by_subsentence.csv', sep=';')
neutral_by_subsentence_switch = pd.read_csv(path + '/neutral_by_subsentence_switch.csv', sep=';')
neutral_by_paraphrased_switch = pd.read_csv(path + '/neutral_by_paraphrased_switch.csv', sep=',')
neutral_by_sentence_continuation = pd.read_csv(path + '/neutral_by_sentence_continuation.csv', sep=';')
contradiction_by_different_noun = pd.read_csv(path + '/contradiction_by_different_noun.csv', sep=';')
neutral_by_non_contradicting_verb = pd.read_csv(path + '/neutral_by_non_contradicting_verb.csv', sep=';')

# drop irrelevant rows
contradiction_by_antonym_adj_adv = contradiction_by_antonym_adj_adv.drop('Unnamed: 0', axis=1)
contradiction_by_different_contents = contradiction_by_different_contents.drop('Unnamed: 0', axis=1)
contradiction_by_different_verbs = contradiction_by_different_verbs.drop('Unnamed: 0', axis=1)
contradiction_by_negation = contradiction_by_negation.drop('Unnamed: 0', axis=1)
neutral_by_modifier = neutral_by_modifier.drop('Unnamed: 0', axis=1)
contradiction_by_number_substitution = contradiction_by_number_substitution.drop('Unnamed: 0', axis=1)
contradiction_by_different_object = contradiction_by_different_object.drop('Unnamed: 0', axis=1)
entailment_by_paraphrasing = entailment_by_paraphrasing.drop('Unnamed: 0', axis=1)
entailment_by_subsentence = entailment_by_subsentence.drop('Unnamed: 0', axis=1)
neutral_by_subsentence_switch = neutral_by_subsentence_switch.drop('Unnamed: 0', axis=1)
neutral_by_paraphrased_switch = neutral_by_paraphrased_switch.drop('Unnamed: 0', axis=1)
neutral_by_sentence_continuation = neutral_by_sentence_continuation.drop('Unnamed: 0', axis=1)
contradiction_by_different_noun = contradiction_by_different_noun.drop('Unnamed: 0', axis=1)
neutral_by_non_contradicting_verb = neutral_by_non_contradicting_verb.drop('Unnamed: 0', axis=1)

# show length of all dataframes
print('contradiction_by_antonym_adj_adv:     ', len(contradiction_by_antonym_adj_adv))
print('contradiction_by_different_contents:  ', len(contradiction_by_different_contents))
print('contradiction_by_different_verbs:     ', len(contradiction_by_different_verbs))
print('contradiction_by_negation:            ', len(contradiction_by_negation))
print('neutral_by_modifier:                  ', len(neutral_by_modifier))
print('contradiction_by_number_substitution: ', len(contradiction_by_number_substitution))
print('contradiction_by_different_object:    ', len(contradiction_by_different_object))
print('entailment_by_paraphrasing:           ', len(entailment_by_paraphrasing))
print('entailment_by_subsentence:            ', len(entailment_by_subsentence))
print('neutral_by_subsentence_switch:        ', len(neutral_by_subsentence_switch))
print('neutral_by_paraphrased_switch:        ', len(neutral_by_paraphrased_switch))
print('neutral_by_sentence_continuation:     ', len(neutral_by_sentence_continuation))
print('contradiction_by_different_noun:      ', len(contradiction_by_different_noun))
print('neutral_by_non_contradicting_verb:    ', len(neutral_by_non_contradicting_verb))

# join all dataframes into one big dataframe
first_join = pd.merge(contradiction_by_negation, contradiction_by_number_substitution, how='left', on=['sent'])
second_join = pd.merge(first_join, contradiction_by_antonym_adj_adv, how='left', on=['sent'])
third_join = pd.merge(second_join, contradiction_by_different_object, how='left', on=['sent'])
fourth_join = pd.merge(third_join, neutral_by_modifier, how='left', on=['sent'])
fifth_join = pd.merge(fourth_join, entailment_by_subsentence, how='left', on=['sent'])
sixth_join = pd.merge(fifth_join, entailment_by_paraphrasing, how='left', on=['sent'])
seventh_join = pd.merge(sixth_join, contradiction_by_different_contents, how='left', on=['sent'])
eights_join = pd.merge(seventh_join, contradiction_by_different_verbs, how='left', on=['sent'])
ninth_join = pd.merge(eights_join, neutral_by_sentence_continuation, how='left', on=['sent'])
tenth_join = pd.merge(ninth_join, contradiction_by_different_noun, how='left', on=['sent'])
eleventh_join = pd.merge(tenth_join, neutral_by_non_contradicting_verb, how='left', on=['sent'])
# concatenate the "switched_sub_sentences"
frames = [eleventh_join, neutral_by_subsentence_switch]
result = pd.concat(frames)
result = result.reset_index()
# concatenate the "neutral_by_paraphrase_switches"
frames = [result, neutral_by_paraphrased_switch]
result = pd.concat(frames)
result = result.reset_index(drop=True)

# show all columns
for col in result.columns:
    print(col)

# unpivot the dataframe to create the final training dataset in 3-column-format
unpivoted = pd.melt(result, id_vars=['sent'], value_vars=['negation_1',
                                                          'negation_2',
                                                          'negation_3',
                                                          'negation_4',
                                                          'negation_5',
                                                          'negation_6',
                                                          'number_subst_1',
                                                          'number_subst_2',
                                                          'number_subst_3',
                                                          'number_subst_4',
                                                          'number_subst_5',
                                                          'number_subst_6',
                                                          'number_subst_7',
                                                          'number_subst_8',
                                                          'number_subst_9',
                                                          'number_subst_10',
                                                          'number_subst_11',
                                                          'number_subst_12',
                                                          'number_subst_13',
                                                          'number_subst_14',
                                                          'number_subst_15',
                                                          'number_subst_16',
                                                          'number_subst_17',
                                                          'number_subst_18',
                                                          'number_subst_19',
                                                          'number_subst_20',
                                                          'number_subst_21',
                                                          'number_subst_22',
                                                          'number_subst_23',
                                                          'number_subst_24',
                                                          'number_subst_25',
                                                          'number_subst_26',
                                                          'number_subst_27',
                                                          'adj_antonym_1',
                                                          'adj_antonym_2',
                                                          'adj_antonym_3',
                                                          'adj_antonym_4',
                                                          'adj_antonym_5',
                                                          'adj_antonym_6',
                                                          'new_obj_1',
                                                          'new_obj_2',
                                                          'new_obj_3',
                                                          'new_obj_4',
                                                          'new_obj_5',
                                                          'neutral_modifier_1',
                                                          'neutral_modifier_2',
                                                          'neutral_modifier_3',
                                                          'neutral_modifier_4',
                                                          'neutral_modifier_5',
                                                          'neutral_modifier_6',
                                                          'neutral_modifier_7',
                                                          'neutral_modifier_8',
                                                          'neutral_modifier_9',
                                                          'neutral_modifier_10',
                                                          'neutral_modifier_11',
                                                          'sub_sent_1',
                                                          'paraphrased_1',
                                                          'paraphrased_2',
                                                          'paraphrased_3',
                                                          'paraphrased_4',
                                                          'paraphrased_5',
                                                          'paraphrased_6',
                                                          'paraphrased_7',
                                                          'paraphrased_8',
                                                          'paraphrased_9',
                                                          'paraphrased_10',
                                                          'different_content_1',
                                                          'different_content_2',
                                                          'different_content_3',
                                                          'different_verb_1',
                                                          'different_verb_2',
                                                          'different_verb_3',
                                                          'different_verb_4',
                                                          'different_verb_5',
                                                          'different_verb_6',
                                                          'different_verb_7',
                                                          'neutral_by_GPT_1',
                                                          'neutral_by_GPT_2',
                                                          'neutral_by_GPT_3',
                                                          'neutral_by_GPT_4',
                                                          'neutral_by_GPT_5',
                                                          'neutral_by_GPT_6',
                                                          'neutral_by_GPT_7',
                                                          'neutral_by_GPT_8',
                                                          'neutral_by_GPT_9',
                                                          'neutral_by_GPT_10',
                                                          'neutral_by_GPT_11',
                                                          'neutral_by_GPT_12',
                                                          'neutral_by_GPT_13',
                                                          'neutral_by_GPT_14',
                                                          'neutral_by_GPT_15',
                                                          'neutral_by_GPT_16',
                                                          'neutral_by_GPT_17',
                                                          'neutral_by_GPT_18',
                                                          'different_noun_1',
                                                          'different_noun_2',
                                                          'different_noun_3',
                                                          'different_noun_4',
                                                          'different_noun_5',
                                                          'different_noun_6',
                                                          'non_contradicting_verb_1',
                                                          'non_contradicting_verb_2',
                                                          'non_contradicting_verb_3',
                                                          'non_contradicting_verb_4',
                                                          'non_contradicting_verb_5',
                                                          'neutral_by_switching_1',
                                                          'neutral_by_switching_2',
                                                          'neutral_by_switching_3',
                                                          'neutral_by_switching_4',
                                                          'neutral_by_switching_5',
                                                          'neutral_by_switching_6',
                                                          'neutral_by_paraphraze_switch'], ignore_index=False)


# clean the sentence columns
def data_cleaning(unpivoted):
    # sort by sentence-column, remove nan-values and set the final column names
    print('length before cleaning: ', len(unpivoted))
    unpivoted = unpivoted.sort_values(by=['sent'])
    unpivoted = unpivoted.dropna()
    unpivoted = unpivoted.rename(columns={"sent": "sentence1", "value": "sentence2", "variable": "gold_label"})
    unpivoted = unpivoted[['sentence1', 'sentence2', 'gold_label']]
    unpivoted = unpivoted.reset_index()
    unpivoted.drop('index', axis=1, inplace=True)
    print('length after cleaning: ', len(unpivoted))
    print('final column names: ', list(unpivoted.columns.values))

    # remove invalid sentences
    counter = 0
    deletions = 0
    unpivoted['del'] = 0
    for index, row in unpivoted.iterrows():
        sent1_ct = str(row['sentence1']).count('(')
        sent2_ct = str(row['sentence2']).count('(')
        if sent1_ct > 2 or sent2_ct > 2:
            deletions += 1
            unpivoted.at[index, 'del'] = 1
        if counter % 5000 == 0:
            print('deletions: ', deletions)
        counter += 1
    unpivoted = unpivoted.drop(unpivoted[unpivoted['del'] == 1].index)
    unpivoted = unpivoted.drop('del', 1)
    print('length after cleaning: ', len(unpivoted))

    # repair sentence, that start with a comma
    sents_row = []
    for index, row in unpivoted.iterrows():
        if row['sentence1'][:2] == ', ':
            sentence1_cl = row['sentence1'][2:]
            sentence1_cl = sentence1_cl[0].upper() + sentence1_cl[1:]
            sents = [sentence1_cl, row['sentence2'], row['gold_label']]
        else:
            if '((' in row['sentence1'] or '))' in row['sentence1'] or \
                    '((' in row['sentence2'] or '))' in row['sentence2']:
                print('(1) row removed!')
            if len(row['sentence1']) < 10 or len(row['sentence2']) < 10:
                print('(2) row removed!')
            else:
                sents = [row['sentence1'], row['sentence2'], row['gold_label']]
        sents_row.append(sents)
        if index % 5000 == 0:
            print('processed sentences', index)

    unpivoted = pd.DataFrame(sents_row, columns=['sentence1', 'sentence2', 'gold_label'])
    # export data set with original label names
    unpivoted.to_csv('output/final_training_set_original_label.csv', index=False)
    return unpivoted



# rename the target/label-classes
def rename_label_classes(unpivoted):
    # rename the target class of the "negated-sentences" to "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'negation_1', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'negation_2', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'negation_3', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'negation_4', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'negation_5', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'negation_6', "gold_label"] = "contradiction"
    # rename the target class of the "switched-number-sentences" to "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_1', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_2', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_3', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_4', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_5', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_6', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_7', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_8', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_9', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_10', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_11', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_12', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_13', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_14', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_15', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_16', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_17', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_18', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_19', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_20', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_21', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_22', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_23', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_24', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_25', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_26', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'number_subst_27', "gold_label"] = "contradiction"
    # rename the target class of the "adjective/adverbs-antonyms-sentences" to "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'adj_antonym_1', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'adj_antonym_2', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'adj_antonym_3', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'adj_antonym_4', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'adj_antonym_5', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'adj_antonym_6', "gold_label"] = "contradiction"
    # rename the target class of the "different-noun-sentences" to "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_noun_1', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_noun_2', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_noun_3', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_noun_4', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_noun_5', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_noun_6', "gold_label"] = "contradiction"
    # rename the target class of the "different-content-sentences" to "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_content_1', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_content_2', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_content_3', "gold_label"] = "contradiction"
    # rename the target class of the "different-verb-sentences" to "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_verb_1', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_verb_2', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_verb_3', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_verb_4', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_verb_5', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_verb_6', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'different_verb_7', "gold_label"] = "contradiction"
    # rename the target class of the "new-object-sentences" to "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'new_obj_1', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'new_obj_2', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'new_obj_3', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'new_obj_4', "gold_label"] = "contradiction"
    unpivoted.loc[unpivoted.gold_label == 'new_obj_5', "gold_label"] = "contradiction"
    # rename the target class of the "neutral-modifier-sentences" to "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_1', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_2', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_3', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_4', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_5', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_6', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_7', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_8', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_9', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_10', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_modifier_11', "gold_label"] = "neutral"
    # rename the target class of the "neutral-by-switching-sentences" to "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_switching_1', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_switching_2', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_switching_3', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_switching_4', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_switching_5', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_switching_6', "gold_label"] = "neutral"
    # rename the target class of the "neutral-by-continuation-sentences" to "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_1', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_2', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_3', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_4', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_5', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_6', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_7', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_8', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_9', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_10', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_11', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_12', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_13', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_14', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_15', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_16', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_17', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_GPT_18', "gold_label"] = "neutral"
    # rename the target class of the "neutral-by-non-contradicting-verb-sentences" to "neutral"
    unpivoted.loc[unpivoted.gold_label == 'non_contradicting_verb_1', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'non_contradicting_verb_2', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'non_contradicting_verb_3', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'non_contradicting_verb_4', "gold_label"] = "neutral"
    unpivoted.loc[unpivoted.gold_label == 'non_contradicting_verb_5', "gold_label"] = "neutral"
    # rename the target class of the "neutral-by-paraphrase-switch-sentences" to "neutral"
    unpivoted.loc[unpivoted.gold_label == 'neutral_by_paraphraze_switch', "gold_label"] = "neutral"
    # rename the target class of the "neutral-by-sub-sentences" to "entailment"
    unpivoted.loc[unpivoted.gold_label == 'sub_sent_1', "gold_label"] = "entailment"
    # rename the target class of the "neutral-by-paraphrased-sentences" to "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_1', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_2', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_3', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_4', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_5', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_6', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_7', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_8', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_9', "gold_label"] = "entailment"
    unpivoted.loc[unpivoted.gold_label == 'paraphrased_10', "gold_label"] = "entailment"
    return unpivoted


# clean sentences and rename target classes
unpivoted = data_cleaning(unpivoted)
unpivoted = rename_label_classes(unpivoted)

# remove duplicates
unpivoted_cl = unpivoted.drop_duplicates()
unpivoted_cl = unpivoted_cl.sort_values(['sentence1', 'sentence2', 'gold_label'], ascending=False)
unpivoted_cl = unpivoted_cl.drop_duplicates(subset=["sentence1", "sentence2"], keep='first')
print('unpivoted length after removing duplicates (2): ', len(unpivoted_cl))

# count occurrences of the 3 target classes (neutral | contradiction | entailment)
occur = unpivoted_cl.groupby(['gold_label']).size()
print('occurrence of the 3 target classes:')
print(occur)

# export final data set to disc
unpivoted_cl.to_csv('output/final_training_set.csv', index=False)
