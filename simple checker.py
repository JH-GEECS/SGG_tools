import spellchecker
import pandas as pd
import os
import glob


spell = spellchecker.SpellChecker()

all_csv = glob.glob(os.path.join(r'Z:\assistant\assistant_deploy\obj_det_anot', '*.csv'))

error_list = []

for idx, each_csv in enumerate(all_csv):
    print(idx, 'start', os.path.basename(each_csv))
    df = pd.read_csv(each_csv)
    word_list = df['class'].unique().tolist()
    word_list_spell_checked = []

    for each_class in word_list:
        corrected_word = []
        for each_word in each_class.split(' '):
            corrected_word.append(spell.correction(each_word))
        correct_word = ' '.join(corrected_word)
        word_list_spell_checked.append(correct_word)

    if word_list != word_list_spell_checked:
        error_list.append(each_csv)

print(error_list)

