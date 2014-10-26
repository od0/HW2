from collections import deque

#import scan
import utils
#import numpy as np
#from sklearn import tree


class DecisionTree(object):
    def __init__(self, review_samples, feature_set):
        # node_label takes the values 0, 1, None. If has the values 0 or 1, then this is a leaf
        self.node_label = None
        self.left = None
        self.right = None
        self.split_word = None
        self.decision(review_samples, feature_set)
        #self.split(review_set, features)

    def is_label(self):
        return self.node_label == 0 or self.node_label == 1

    def _get_label(self, review_set):
        max_label = max([sample.rating for sample in review_set])
        if not 1 >= max_label >= 0:
            raise AttributeError("Ivalid label: %d. Decision tree label must be 0 or 1." % max_label)
        self.node_label = max_label

    def decision(self, review_set, features):
        entropy = utils.entropy([sample.rating for sample in review_set])

        if entropy < 1:
            full_set_length = len(review_set)
            # max_info_gain holds info gain
            max_info_gain = (0, None, None)
            for word in features:
                left_set, right_set = DecisionTree.split(word, review_set)
                info_gain = utils.information_gain(entropy, full_set_length, left_set, right_set)
                if info_gain > max_info_gain[0]:
                    max_info_gain = (info_gain, word, left_set, right_set)
                #info_gain_features[word] = (info_gain, left_set, right_set)

            # Exclude this word/attribute
            self.split_word = max_info_gain[1]
            features.remove(self.split_word)
            self.left = DecisionTree(max_info_gain[2], features)
            self.right = DecisionTree(max_info_gain[3], features)
        else:
            self._get_label(review_set)

    @classmethod
    def split(cls, word, review_set):
        left_indices = [i for (i, sample) in enumerate(review_set) if word in sample.word_list]
        left = [sample.rating for (i, sample) in enumerate(review_set) if i in left_indices]
        right = [sample.rating for (i, sample) in enumerate(review_set) if i not in left_indices]
        return left, right


    #def go(self, data):
        #if self.node_label is None:
            #return self.node_label
        #return self.go(self.decision(data))


# http://en.wikipedia.org/wiki/ID3_algorithm
def train(review_samples, feature_set):
    decision_tree = DecisionTree(review_samples, feature_set)
    return decision_tree


def test(review_samples, decision_tree):
    raise NotImplementedError

