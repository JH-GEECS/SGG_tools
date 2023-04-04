import csv
import json
import pandas as pd

data = [['Predicate', 'Preposition', 'Classification'],
        ['and', 'None', 'Other'],
        ['says', 'None', 'Other'],
        ['belonging to', 'None', 'Other'],
        ['over', 'over', 'Location'],
        ['parked on', 'on', 'Location'],
        ['growing on', 'on', 'Location'],
        ['standing on', 'on', 'Location'],
        ['made of', 'of', 'Composition'],
        ['part of', 'of', 'Composition'],
        ['attached to', 'to', 'Attachment'],
        ['at', 'at', 'Location'],
        ['in', 'in', 'Location'],
        ['hanging from', 'from', 'Attachment'],
        ['wears', 'None', 'Other'],
        ['in front of', 'of', 'Location'],
        ['from', 'from', 'Source'],
        ['for', 'for', 'Purpose'],
        ['lying on', 'on', 'Location'],
        ['to', 'to', 'Direction'],
        ['behind', 'behind', 'Location'],
        ['flying in', 'in', 'Location'],
        ['looking at', 'at', 'Location'],
        ['on back of', 'on', 'Location'],
        ['holding', 'None', 'Other'],
        ['under', 'under', 'Location'],
        ['laying on', 'on', 'Location'],
        ['riding', 'None', 'Other'],
        ['has', 'None', 'Other'],
        ['across', 'across', 'Direction'],
        ['wearing', 'None', 'Other'],
        ['walking on', 'on', 'Location'],
        ['eating', 'None', 'Other'],
        ['above', 'above', 'Location'],
        ['watching', 'None', 'Other'],
        ['walking in', 'in', 'Location'],
        ['sitting on', 'on', 'Location'],
        ['between', 'between', 'Location'],
        ['covered in', 'in', 'Location'],
        ['carrying', 'None', 'Other'],
        ['using', 'None', 'Other'],
        ['along', 'along', 'Direction'],
        ['with', 'with', 'Accompaniment'],
        ['on', 'on', 'Location'],
        ['covering', 'None', 'Other'],
        ['of', 'of', 'Possession'],
        ['against', 'against', 'Contact'],
        ['mounted on', 'on', 'Location'],
        ['near', 'near', 'Location'],
        ['painted on', 'on', 'Location'],
        ['playing', 'None', 'Other']]

with open('predicates.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)\

df = pd.read_csv('result.csv')

stat = {}
classification = {}
for word in df['Preposition'].unique():
        count = len(df[df['Preposition'] == word])
        words_list = df[df['Preposition'] == word]['Predicate'].tolist()
        stat[word] = count
        classification[word] = words_list

with open("predicate_tree_classification.json", "w") as f:
    json.dump(classification, f)


test = 1