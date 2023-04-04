# 이호준 annotated file의 철자 교정을 시행한다.
# 어떠한 오타가 있는지 확인한다.
import spellchecker
import pandas as pd
import os
import glob
import time

def spell_checker(original, processed, debug_dir):

    spell = spellchecker.SpellChecker()
    obj_annot_csv = glob.glob(os.path.join(original, '*.csv'))
    os.makedirs(processed, exist_ok=True)
    word_counter = {}

    for each_csv in obj_annot_csv:
        start = time.time()
        print(f'processing {os.path.basename(each_csv)}')
        df = pd.read_csv(each_csv)
        # 여기는 row iterations으로 가야함.
        for idx, row in df.iterrows():
            each_class = row['class']
            word_counter[each_class] = word_counter.get(each_class, 0) + 1
            corrected_word = []
            for each_word in each_class.split(' '):
                corrected_word.append(spell.correction(each_word))
            df.at[idx, 'class'] = ' '.join(corrected_word)

        df.to_csv(os.path.join(processed, os.path.basename(each_csv)), index=False)
        print(f'done @ {time.time() - start}')

    word_statistics = pd.DataFrame(word_counter, index=[0])
    word_statistics.to_csv(os.path.join(debug_dir, 'word_statistics.csv'), index=False)

if __name__ == '__main__':
    start_time = time.time()
    original = r'Z:\assistant\assistant_deploy\obj_det_anot'
    processed = r'Z:\assistant\assistant_deploy\obj_det_anot_spc_2'
    debug_dir = r'Z:\assistant\assistant_deploy\debug_result'
    spell_checker(original, processed, debug_dir)
    print(f'done @ {time.time() - start_time}')
