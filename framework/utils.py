from __future__ import division
import math


# decision tree stuff
def entropy(class_set):
    total = len(class_set)
    positives = len([r for r in class_set if r])
    negatives = len([r for r in class_set if not r])
    if positives and negatives:
        return - (
            positives/total * math.log(positives/total) +
            negatives/total * math.log(negatives/total)
        )
    else:
        # If either positives or negatives is 0, this is a pure set (entropy=0)
        return 0


def information_gain(base_entropy, full_set_length, subset_left, subset_right):
    #H_of_set, set_length = entropy(full_set), len(full_set)
    H_of_left, left_length = entropy(subset_left), len(subset_left)
    H_of_right, right_length = entropy(subset_right), len(subset_right)
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
