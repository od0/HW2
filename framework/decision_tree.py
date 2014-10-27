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
    def __init__(self, depth=0, desc=None):
        # node_label takes the values 0, 1, None. If has the values 0 or 1, then this is a leaf
        self.node_label = None
        self.left = None
        self.right = None
        self.depth = depth
        self.split_word = None
        if desc:
            self.desc = desc
        #self.decision(review_samples, feature_set, depth)

    def is_label(self):
        return not self.node_label is None

    def set_label(self, review_set):
        num_positive = len([sample.rating for sample in review_set if sample.rating == 1])
        if num_positive / len(review_set) > 0.5:
            self.node_label = 1
        else:
            self.node_label = 0
        logging.debug('Set label=%d for %s at depth %d with %d items remaining in reviewset' % (
            self.node_label, self.desc, self.depth, len(review_set)
        ))

    #@profile
    def decision(self, review_set, features):
        entropy = utils.entropy(np.array([sample.rating for sample in review_set]))

        if 1 > entropy > 0 and len(features) > 0:
            full_set_length = len(review_set)
            # Max Info Gain holds 4 values:
            #   (info_gain, word, left_set, right_set)
            max_info_gain = (0, None, None, None)

            for word in features:
                left_labels, right_labels, left_indices, right_indices = DecisionTree.split(word, review_set)
                info_gain = utils.information_gain(entropy, full_set_length, left_labels, right_labels)
                if info_gain > max_info_gain[0]:
                    left_set = [review_set[i] for i in left_indices]
                    right_set = [review_set[i] for i in right_indices]
                    max_info_gain = (info_gain, word, left_set, right_set)

            if max_info_gain[0] > 0:
                self.split_word = max_info_gain[1]
                # Exclude this word/attribute
                features.remove(self.split_word)
                logging.debug('Splitting on %s (info gain %0.5f, branch size %d) (node_label is %s)' % (
                    self.split_word, max_info_gain[0], len(review_set), self.node_label
                ))

                return max_info_gain

                # Create 2 new DTrees
                '''self.left = DecisionTree(
                    max_info_gain[2], features,
                    depth=depth+1,
                    desc='\"%s\" appears' % self.split_word
                )
                self.right = DecisionTree(
                    max_info_gain[3], features,
                    depth=depth+1,
                    desc='\"%s\" does not appear' % self.split_word
                )'''

            else:
                # None of the words provided info gain > 0. No further splitting is possible.
                logging.debug('No measurable info gain for any of the %d remaining words' % len(features))
                self.set_label(review_set)

        else:
            self.set_label(review_set)

        return None

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
    result = decision_tree.decision(review_samples, feature_set)
    if result:
        extend_tree(result, tree_queue, decision_tree)

    next_rec = None
    while len(tree_queue) > 0:
         next_rec = tree_queue.popleft()
         decision_tree = next_rec.get('tree')
         result = decision_tree.decision(next_rec.get('reviews_set'), feature_set)
         if result:
            extend_tree(result, tree_queue, decision_tree)

    return root


def extend_tree(result, tree_queue, decision_tree):
    #decision_tree.split_word = result[1]
    decision_tree.left = DecisionTree(
        depth=decision_tree.depth+1,
        desc='\"%s\" appears' % decision_tree.split_word
        )
    tree_queue.append({
        'tree': decision_tree.left, 
        'reviews_set': result[2]
        })

    decision_tree.right = DecisionTree(
        depth=decision_tree.depth+1,
        desc='\"%s\" does not appear' % decision_tree.split_word
        )
    tree_queue.append({
        'tree': decision_tree.right,
        'reviews_set': result[3]
        })


def test(review_samples, decision_tree):
    if not decision_tree.is_label():
        left_indices = frozenset([
            i for (i, sample) in enumerate(review_samples) 
            if decision_tree.split_word in sample.word_set
            ])
        right_indices = frozenset(range(len(review_samples))).difference(left_indices)
        #left_indices = [i for (i, sample) in enumerate(review_samples)
                        #if decision_tree.split_word in sample.word_set]

        #left_set = [sample for (i, sample) in enumerate(review_samples) if i in left_indices]
        #right_set = [sample for (i, sample) in enumerate(review_samples) if i not in left_indices]
        left_set = [review_samples[i] for i in left_indices]
        right_set = [review_samples[i] for i in right_indices]
        logging.debug('Test split on word=%s (left=%d, right=%d' % (
            decision_tree.split_word, len(left_set), len(right_set)
            ))
        if left_set:
            test(left_set, decision_tree.left)
        if right_set:
            test(right_set, decision_tree.right)
    else:
        # Found label
        logging.debug('Applying label=%d for %d total samples.' % (decision_tree.node_label, len(review_samples)))
        for sample in review_samples:
            sample.predicted_rating = decision_tree.node_label
        return
