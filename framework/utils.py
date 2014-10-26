from __future__ import division
import math

import numpy as np


# decision tree stuff
#@profile
def entropy(class_set):
    total = class_set.size
    positives = np.flatnonzero(class_set).size
    negatives = total - positives
    if positives and negatives:
        return - (
            positives/total * math.log(positives/total) +
            negatives/total * math.log(negatives/total)
        )
    else:
        # If either positives or negatives is 0, this is a pure set (entropy=0)
        return 0


#@profile
def information_gain(base_entropy, full_set_length, subset_left, subset_right):
    H_of_left, left_length = entropy(subset_left), subset_left.size
    H_of_right, right_length = entropy(subset_right), subset_right.size
    return base_entropy - (
        (left_length / full_set_length * H_of_left) +
        (right_length / full_set_length * H_of_right)
    )


# natural language processing stuff
def freq(lst):
    freq = {}
    length = len(lst)
    for ele in lst:
        if ele not in freq:
            freq[ele] = 0
        freq[ele] += 1
    return (freq, length)


def get_unigram(review):
    return freq(review.split())


def get_unigram_list(review):
    return get_unigram(review)[0].keys()
