import logging
import datetime
from collections import deque

import numpy as np

import config
import utils


logStart = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    filename='logs/%s_decision.log' % logStart,
    level=logging.DEBUG if config.RUN_FILTER['debug'] else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class DecisionTree(object):
    def __init__(self, depth=0, old_desc=list(), desc=None):
        # node_label takes the values 0, 1, None. 
        # If has the values 0 or 1, then this is a leaf
        self.node_label = None
        self.left = None
        self.right = None

        self.depth = depth
        self.split_word = None
        if old_desc:
            self.desc = list(old_desc)
        else:
            self.desc = list()
        if desc:
            self.desc.append(desc)

    def is_label(self):
        return not self.node_label is None

    def set_label(self, sub_set):
        num_positive = len([sample.rating for sample in sub_set if sample.rating])
        if num_positive >= (len(sub_set) - num_positive):
            self.node_label = 1
        else:
            self.node_label = 0
        logging.debug('Set label=%d for %d reviews over %s at depth %d. (positive=%d, negative=%d)' % (
            self.node_label, len(sub_set), self.desc, self.depth,
            num_positive,
            len(sub_set) - num_positive
        ))

    #@profile
    def decision(self, review_set, features):
        entropy = utils.entropy(np.array([sample.rating for sample in review_set]))

        left_set, right_set = None, None
        if 1 > entropy > 0 and len(features) > 0:
            max_info_gain, split_word, left_set, right_set = (
                DecisionTree.maximize_info_gain(review_set, entropy, features)
            )

            if max_info_gain > 0:
                self.split_word = split_word
                # Exclude this word/attribute from future splits
                features.remove(self.split_word)

                logging.debug('Splitting on %s (info gain %0.5f, total:%d, left:%d, right:%d)' % (
                    self.split_word, max_info_gain,
                    len(review_set), len(left_set), len(right_set)
                ))

            else:
                # None of the words provided info gain > 0. 
                # No further splitting is possible.
                logging.debug('No measurable info gain for any of the %d remaining words' % len(features))
                self.set_label(review_set)
        else:
            self.set_label(review_set)

        return left_set, right_set

    @classmethod
    def maximize_info_gain(cls, review_set, entropy, features):
        full_set_length = len(review_set)

        max_info_gain = 0
        split_word, left_set, right_set = None, None, None

        for word in features:
            left_labels, right_labels, left_indices, right_indices = DecisionTree.split(word, review_set)
            info_gain = utils.information_gain(entropy, full_set_length, left_labels, right_labels)
            if info_gain > max_info_gain:
                max_info_gain = info_gain
                split_word = word
                left_set = [review_set[i] for i in left_indices]
                right_set = [review_set[i] for i in right_indices]
                #max_info_gain = (info_gain, word, left_set, right_set)

        return max_info_gain, split_word, left_set, right_set

    @classmethod
    #@profile
    def split(cls, word, review_set):
        left_indices = frozenset([i for (i, sample) in enumerate(review_set) if word in sample.word_set])
        right_indices = frozenset(range(len(review_set))).difference(left_indices)
        left = np.array([review_set[i].rating for i in left_indices])
        right = np.array([review_set[i].rating for i in right_indices])
        return left, right, left_indices, right_indices

    def __repr__(self):
        return "<DecisionTree()> %s" % (self.desc if self.desc else '')


# http://en.wikipedia.org/wiki/ID3_algorithm
#@profile
def train(review_samples, feature_set):
    #decision_tree = DecisionTree(review_samples, feature_set)
    tree_queue = deque()
    decision_tree = DecisionTree()
    root = decision_tree
    left_set, right_set = decision_tree.decision(review_samples, feature_set)
    if left_set or right_set:
        extend_tree(tree_queue, decision_tree, left_set, right_set)

    while len(tree_queue) > 0:
        one_item = tree_queue.popleft()
        decision_tree = one_item.get('tree')
        sub_set = one_item.get('sub_set')
        left_set, right_set = decision_tree.decision(sub_set, feature_set)
        if left_set or right_set:
            extend_tree(tree_queue, decision_tree, left_set, right_set)

    return root


def extend_tree(tree_queue, decision_tree, left_set, right_set):
    decision_tree.left = DecisionTree(
        depth=decision_tree.depth+1,
        old_desc=decision_tree.desc,
        desc='\"%s\" appears' % decision_tree.split_word
    )
    tree_queue.append({
        'tree': decision_tree.left, 
        'sub_set': left_set
    })

    decision_tree.right = DecisionTree(
        depth=decision_tree.depth+1,
        old_desc=decision_tree.desc,
        desc='\"%s\" missing' % decision_tree.split_word
    )
    tree_queue.append({
        'tree': decision_tree.right,
        'sub_set': right_set
    })


def test(review_samples, decision_tree):
    if not decision_tree.is_label():
        left_indices = frozenset([
            i for (i, sample) in enumerate(review_samples) 
            if decision_tree.split_word in sample.word_set
        ])
        right_indices = frozenset(range(len(review_samples))).difference(left_indices)
        left_set = [review_samples[i] for i in left_indices]
        right_set = [review_samples[i] for i in right_indices]

        if left_set:
            test(left_set, decision_tree.left)
        if right_set:
            test(right_set, decision_tree.right)

        logging.debug('Test split on word=%s (left=%d, right=%d' % (
            decision_tree.split_word, len(left_set), len(right_set)
        ))

    else:
        # Found label
        logging.debug('Applying label=%d for %d total samples.' % (decision_tree.node_label, len(review_samples)))
        for sample in review_samples:
            sample.predicted_rating = decision_tree.node_label
        return
