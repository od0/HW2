import logging
import datetime
from collections import deque
from multiprocessing import Process, Queue
import os

import numpy as np

import config
import utils

if not os.path.exists('logs'):
    os.mkdir('logs')
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
        logging.debug('Set label=%d for %d reviews over %s at depth %d. '
            '(positive=%d, negative=%d)' % (
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

                logging.debug('Splitting on %s (info gain %0.5f, total:%d, '
                    'left:%d, right:%d)' % (
                    self.split_word, max_info_gain,
                    len(review_set), len(left_set), len(right_set)
                ))

            else:
                # None of the words provided info gain > 0. 
                # No further splitting is possible.
                logging.debug('No measurable info gain for any of the %d '
                    'remaining words' % len(features))
                self.set_label(review_set)
        else:
            self.set_label(review_set)

        return left_set, right_set

    @classmethod
    def parallel_split(cls, split_queue, ig_queue, review_set):
        while True:
            word = split_queue.get()
            if (word != 'STOP'):
                result = DecisionTree.split(word, review_set)
                ig_queue.put((result, word))
            else:
                ig_queue.put('STOP')
                break
        return

    @classmethod
    def parallel_ig(cls, queue, length, entropy, result_queue):
        while True:
            item = queue.get()
            if (item != 'STOP'):
                try:
                    left_labels = item[0][0]
                    left_indices = item[0][2]
                    right_labels = item[0][1]
                    right_indices = item[0][3]
                    word = item[1]
                    info_gain = utils.information_gain(entropy, length, left_labels, right_labels)
                    one_result = {
                        'gain': info_gain, 
                        'word': word,
                        #'left': left_indices
                        #'right': right_indices
                    }
                    result_queue.put(one_result)
                except:
                    pass
            else:
                break

    @classmethod
    def maximize_info_gain(cls, review_set, entropy, features):
        # Find the word in features with the highest information gain
        split_word, left_set, right_set = None, None, None
        full_set_length = len(review_set)
        
        split_queue = Queue()
        ig_queue = Queue()
        result_queue = Queue()

        split_process = Process(target=DecisionTree.parallel_split, 
            args=(split_queue, ig_queue, review_set))
        split_process.daemon = True
        ig_process = Process(target=DecisionTree.parallel_ig, 
            args=(ig_queue, full_set_length, entropy, result_queue))
        ig_process.daemon = True

        split_process.start()
        ig_process.start()

        for word in features:
            split_queue.put(word)
        split_queue.put('STOP')

        split_process.join()
        ig_process.join()

        max_info_gain = {'gain': 0, 'word': ''}
        while not result_queue.empty():
            item = result_queue.get()
            if item.get('gain') > max_info_gain.get('gain'):
                max_info_gain = item
        split_result = DecisionTree.split(max_info_gain.get('word'), review_set)
        left_indices = split_result[2]
        right_indices = split_result[3]
        left_set = [review_set[i] for i in left_indices]
        right_set = [review_set[i] for i in right_indices]
        return max_info_gain.get('gain'), max_info_gain.get('word'), left_set, right_set

    @classmethod
    #@profile
    def split(cls, word, review_set):
        left_indices = frozenset(
            [i for (i, sample) in enumerate(review_set) if word in sample.word_set]
        )
        right_indices = frozenset(range(len(review_set))).difference(left_indices)
        left = np.array([review_set[i].rating for i in left_indices])
        right = np.array([review_set[i].rating for i in right_indices])
        return (left, right, left_indices, right_indices)

    def __repr__(self):
        return "<DecisionTree()> %s" % (self.desc if self.desc else '')


# http://en.wikipedia.org/wiki/ID3_algorithm
#@profile
def train(review_samples, feature_set):
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
    # Adds to new trees to the left and right nodes of the current tree
    # Left trees always "have" the split word
    # Right trees are "missing" the split word
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
        logging.debug('Applying label=%d for %d total samples.' % (
            decision_tree.node_label, len(review_samples))
        )
        for sample in review_samples:
            sample.predicted_rating = decision_tree.node_label
        return
