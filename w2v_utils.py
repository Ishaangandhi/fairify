# Adapted from Coursera Deep Learning MOOC

import numpy as np
from tqdm import tqdm

MAX_WORDS = 100000

def read_glove_vecs(glove_file, bias_files):
    bias_words = ""
    for f in bias_files:
        with open(f, 'r') as myfile:
            bias_words+=myfile.read()

    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        i = 0
        pbar = tqdm(total=MAX_WORDS, desc="Building...")
        for line in f:
            pbar.update(1)
            i+=1
            line = line.strip().split()
            if i > MAX_WORDS and line[0] not in bias_words:
                continue
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)
        pbar.close()
    return words, word_to_vec_map
