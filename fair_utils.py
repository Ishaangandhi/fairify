# Adapted from Coursera Deep Learning MOOC
import numpy as np
from w2v_utils import read_glove_vecs
import requests
from tqdm import tqdm

print("Loading Taboola data...")
r = requests.get("https://us-central1-vision-migration.cloudfunctions.net/la_hacks_2019").json()
list_of_articles = []
buckets = r.get("buckets")
for bucket in tqdm(buckets, desc="Organizing articles..."):
    rollups = bucket.get("report").get("rollups")
    for rollup in rollups:
        articles = rollup.get("top_articles_on_network")
        list_of_articles.extend([[*x][0] for x in articles])

def search_for_article(keyword):
    if keyword in ["brother", "boy", "he", "his", "son", "male", "sister", "man"]:
        return ""
    if keyword == "jew":
        keyword = "israel"

    global list_of_articles
    for article in list_of_articles:
        if keyword in article:
            return article


def cosine_similarity(u, v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by
        the formula above.
    """

    distance = 0.0

    dot = np.dot(u,v)
    norm_u = np.linalg.norm(u)

    norm_v = np.linalg.norm(v)
    cosine_similarity = dot/norm_u/norm_v

    return cosine_similarity


def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____.

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors.

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """

    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    try:
        e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]
    except:
        return "unknown word"

    words = word_to_vec_map.keys()
    max_cosine_sim = -100              # Initialize max_cosine_sim to a large negative number
    best_word = None                   # Initialize best_word with None, it will help keep track of the word to output

    for w in words:
        # to avoid best_word being one of the input words, pass on them.
        if w in [word_a, word_c] :
            continue

        cosine_sim = cosine_similarity(e_b-e_a,word_to_vec_map[w]-e_c)

        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w

    return best_word

def project_down(e, g):
    """
    Returns debiased and biased components of a word
    """

    e_biascomponent = (np.dot(e,g)/(np.linalg.norm(g)**2))*g

    e_debiased = e-e_biascomponent

    return np.linalg.norm(e_debiased), np.linalg.norm(e_biascomponent)



def neutralize(word, g, word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis.
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """

    e = word_to_vec_map[word]

    e_biascomponent = (np.dot(e,g)/(np.linalg.norm(g)**2))*g

    e_debiased = e-e_biascomponent

    return e_debiased


def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor")
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """

    np.seterr(all='raise')
    w1, w2 = pair[0].lower(), pair[1].lower()
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    mu = (e_w1+e_w2)/2.0

    try:
        mu_B = np.dot(mu,bias_axis)/np.linalg.norm(bias_axis)**2*bias_axis
        mu_orth = mu-mu_B

        e_w1B = np.dot(e_w1,bias_axis)/np.linalg.norm(bias_axis)**2*bias_axis
        e_w2B = np.dot(e_w2,bias_axis)/np.linalg.norm(bias_axis)**2*bias_axis

        corrected_e_w1B = (1-np.linalg.norm(mu_orth)**2)**0.5*(e_w1B-mu_B)/np.linalg.norm(e_w1-mu_orth-mu_B)
        corrected_e_w2B = (1-np.linalg.norm(mu_orth)**2)**0.5*(e_w2B-mu_B)/np.linalg.norm(e_w2-mu_orth-mu_B)

        e1 = corrected_e_w1B+mu_orth
        e2 = corrected_e_w2B+mu_orth


        return e1, e2
    except:
        return e_w1, e_w2

def create_analogy(triad, completion, e0, e1, e2, g, word_to_vec_map, is_user):
    # OLD--> a1:a2 :: b1:b2
    analogy = {}
    a1 = triad[0]
    a2 = triad[1]
    b1 = triad[2]
    b2 = completion
    a1x, a1y = project_down(word_to_vec_map[a1], g)
    a2x, a2y = project_down(word_to_vec_map[a2], g)
    b1x, b1y = project_down(word_to_vec_map[b1], g)
    b2x, b2y = project_down(word_to_vec_map[completion], g)

    # coordinates of new vector
    a1xn, a1yn = project_down(e0, g)
    a2xn, a2yn = project_down(e1, g)
    b1xn, b1yn = project_down(e2, g)
    b2xn, b2yn = b2x, b2y

    max_x = max(a1x, a2x, b1x, b2x)
    max_y = max(a1y, a2y, b1y, b2y)

    analogy = {
        "a1": a1,
        "a2": a2,
        "b1": b1,
        "b2": b2,

        "a1x": a1x/max_x,
        "a2x": a2x/max_x,
        "b1x": b1x/max_x,
        "b2x": b2x/max_x,
        "a1y": a1y/max_y,
        "a2y": a2y/max_y,
        "b1y": b1y/max_y,
        "b2y": b2y/max_y,

        "a1xn": a1xn/max_x,
        "a2xn": a2xn/max_x,
        "b1xn": b1xn/max_x,
        "b2xn": b2xn/max_x,
        "a1yn": a1yn/max_y,
        "a2yn": a2yn/max_y,
        "b1yn": b1yn/max_y,
        "b2yn": b2yn/max_y,
        "is_user": is_user,
        "should_fix": False,
        "is_fixed": False,
    }
    if not is_user:
        analogy["taboola_url"] = search_for_article(a1)
    return analogy

def correct_bias(triad, g, word_to_vec_map):
    e1 = neutralize(triad[1], g, word_to_vec_map)
    e0, e2 = equalize((triad[0], triad[2]), g, word_to_vec_map)
    word_to_vec_map[triad[0]] = e0
    word_to_vec_map[triad[1]] = e1
    word_to_vec_map[triad[2]] = e2
    return word_to_vec_map
