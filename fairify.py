from fair_utils import *
from termcolor import colored
import sys
import os
from tqdm import tqdm
import random
from firebase import FirebaseModel
from DataModel import DataModel
from sklearn.manifold import TSNE
import signal
from threading import Thread

TESTING = 0
dm = DataModel()
dm.word_to_vec_map = {}
xeno_triads = []

fb = FirebaseModel(dm)
thread = Thread(target=fb.listen, daemon=True)

def write_word_map(map, filename):
    filename = "0_" + filename
    count = 0
    while os.path.isfile(filename):
        filename = str(count) + "_" + filename[2:]
        count += 1

    for word in tqdm(map, desc="Writing updated model to disk..."):
    # for word in tqdm({"dummy": "data"}, desc="Writing updated model to disk..."):
        with open(filename, 'a') as f:
            e = ' '.join([str(num) for num in map[word]])
            f.write(word + ' ' + e + '\n')


def load_analogies():
    analogies = []
    with open("analogies.txt", 'r') as f:
        for line in f:
            line = line.strip().split()
            if len(line) < 3:
                continue
            analogies.append((line[0], line[1], line[2]))

    return analogies

def generate_triads():
    global xeno_triads
    pairs = []
    extreme_male = []
    extreme_female = []
    triads = []
    with open("gendered_pairs.txt", 'r') as f:
        for line in f:
            line = line.strip().split()
            pairs.append(line)

    with open("extreme_male.txt", 'r') as f:
        for line in f:
            line = line.strip()
            extreme_male.append(line)

    with open("extreme_female.txt", 'r') as f:
        for line in f:
            line = line.strip()
            extreme_female.append(line)

    for pair in pairs:
        for word in extreme_male:
            triads.append((pair[0], word, pair[1]))
        for word in extreme_female:
            triads.append((pair[1], word, pair[0]))

        xeno_pairs = []
        xeno_traits = []
        with open("xeno_pairs.txt", 'r') as f:
            for line in f:
                line = line.strip().split()
                xeno_pairs.append(line)

        with open("xeno_traits.txt", 'r') as f:
            for line in f:
                line = line.strip()
                xeno_traits.append(line)

        for pair in xeno_pairs:
            for word in xeno_traits:
                xeno_triads.append((pair[0], word, pair[1]))

    if TESTING:
        return random.sample(triads,  10)
    return triads


def wait_on_updates():
    print('Listening for fixes. (press Ctrl+C to exit)')
    signal.pause()

def signal_handler(sig, frame):
        print('\n')
        write_word_map(dm.word_to_vec_map, "debiased_model.txt")
        sys.exit(0)


def init_fairify():
    """
        Loads model to be fixed
    """

    # read command line arguments
    filename = sys.argv[1]
    verbose = "--verbose" in sys.argv
    load  = "--load" in sys.argv

    fb.update_name(filename)
    signal.signal(signal.SIGINT, signal_handler)

    print("Loading model at " + colored(filename, "magenta"))
    bias_files = ["gendered_pairs.txt", "extreme_male.txt", "extreme_female.txt"]
    words, dm.word_to_vec_map = read_glove_vecs(filename, bias_files)

    thread.start()

    if load:
        analogies = load_analogies()
    else:
        analogies = generate_triads()

    print(colored("Checking  model for gender bias and xenophobia...", "red"))
    if not verbose:
        pbar = tqdm(total=len(analogies), desc="Analyzing...")

    # vars to calculate model bias score
    good_examples = 0
    total_examples = 0

    zipped_analogies = []
    for i in range(max(len(analogies), len(xeno_triads))):
        if i < len(analogies):
            zipped_analogies.append(analogies[i])
        if i < len(xeno_triads):
            zipped_analogies.append(xeno_triads[i])

    for triad in zipped_analogies:
        completion = complete_analogy(*triad,dm.word_to_vec_map)
        if completion == "unknown word":
            continue
        total_examples+=1
        if completion == triad[1]:
            # good
            good_examples+=1
            if verbose:
                print ('✅ {} -> {} :: {} -> '.format(*triad) + colored(completion, "white", attrs=["underline"]))
            else:
                pbar.update(1)
        else:
            g = dm.word_to_vec_map[triad[0]] - dm.word_to_vec_map[triad[2]] #direction of bias
            e1 = neutralize(triad[1], g, dm.word_to_vec_map)
            e0, e2 = equalize((triad[0], triad[2]), g, dm.word_to_vec_map)
            # create an analogy object to add to firebase
            fb.add_analogy(create_analogy(triad, completion, e0, e1, e2, g,
                        dm.word_to_vec_map, False))
            if verbose:
                print ('❌ {} -> {} :: {} -> '.format(*triad) + colored(completion, "white", attrs=["underline"]))
            else:
                pbar.update(1)
        fb.update_percent(good_examples/total_examples)
    if not verbose:
        pbar.close()

    wait_on_updates()

if __name__ == "__main__":
    init_fairify()
