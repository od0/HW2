import logging
logging.basicConfig(filename='logs/decision.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import utils


class DecisionTree(object):
    def __init__(self, review_samples, feature_set, depth=0, desc=None):
        # node_label takes the values 0, 1, None. If has the values 0 or 1, then this is a leaf
        self.node_label = None
        self.left = None
        self.right = None
        self.depth = depth
        self.split_word = None
        if desc:
            self.desc = desc
        self.decision(review_samples, feature_set, depth)

    def is_label(self):
        return self.node_label == 0 or self.node_label == 1

    def set_label(self, review_set):
        num_positive = len([sample.rating for sample in review_set if sample.rating == 1])
        if num_positive / len(review_set) > 0.5:
            logging.debug('Label=1 where %s at depth %d (%d elements)' % (self, self.depth, len(review_set)))
            self.node_label = 1
        else:
            logging.debug('Label=0 where %s at depth %d (%d elements)' % (self.desc, self.depth, len(review_set)))
            self.node_label = 0

    def decision(self, review_set, features, depth):
        entropy = utils.entropy([sample.rating for sample in review_set])
        if len(review_set) == 2033:
            print

        if 1 > entropy > 0 and len(features) > 0:
            full_set_length = len(review_set)
            # max_info_gain contains (info_gain, word, left_set, right_set)
            max_info_gain = (0, None, None, None)
            for word in features:
                left_labels, right_labels, left_indices = DecisionTree.split(word, review_set)
                info_gain = utils.information_gain(entropy, full_set_length, left_labels, right_labels)
                #logging.debug('\n\t\tWord:%s\n\t\tIG: %0.5f' % (word, info_gain))
                if info_gain > max_info_gain[0]:
                    left_set = [sample for (i, sample) in enumerate(review_set) if i in left_indices]
                    right_set = [sample for (i, sample) in enumerate(review_set) if i not in left_indices]
                    max_info_gain = (info_gain, word, left_set, right_set)
            #try:
            if max_info_gain[0] > 0:
                self.split_word = max_info_gain[1]
                # Exclude this word/attribute
                features.remove(self.split_word)
                logging.debug('Splitting on %s (info gain %0.5f, branch size %d)' % (
                    self.split_word, max_info_gain[0], len(review_set)
                ))
                self.left = DecisionTree(max_info_gain[2], features,
                                         depth=depth+1, desc='%s appears' % self.split_word)
                self.right = DecisionTree(max_info_gain[3], features,
                                          depth=depth+1, desc='%s does not appear' % self.split_word)
            else:
                # None of the words provided info gain > 0
                # Therefore, make this node a label. No further splitting is possible.
                logging.debug('No measurable info gain for any of the %d remaining words' % len(features))
                self.set_label(review_set)
            #except KeyError as e:
                '''num_positive = len([sample.rating for sample in review_set if sample.rating == 1])
                logging.debug("\n\t\tError: %s\n\t\tEntropy: %0.10f\n\t\tNum Reviews: %d"
                              "\n\t\tPositive/Total: %0.10f (%d)"
                              "\n\t\tNegative/Total: %0.10f (%d)"
                              "\n\t\tNum Features: %d\n\t\tMax Info:%s"
                              "\n\t\tDT Desc: %s (%d depth)" % (
                    e, entropy, len(review_set),
                    num_positive / len(review_set), num_positive,
                    (len(review_set) - num_positive) / len(review_set),  (len(review_set) - num_positive),
                    len(features), max_info_gain, self.desc, self.depth
                ))'''
        else:
            self.set_label(review_set)

    @classmethod
    def split(cls, word, review_set):
        left_indices = [i for (i, sample) in enumerate(review_set) if word in sample.word_list]
        left = [sample.rating for (i, sample) in enumerate(review_set) if i in left_indices]
        right = [sample.rating for (i, sample) in enumerate(review_set) if i not in left_indices]
        return left, right, left_indices

    def __repr__(self):
        return "<DecisionTree(%s)>" % (self.desc if self.desc else '')


# http://en.wikipedia.org/wiki/ID3_algorithm
def train(review_samples, feature_set):
    decision_tree = DecisionTree(review_samples, feature_set)
    return decision_tree


def test(review_samples, decision_tree):
    if decision_tree.is_label or not review_samples:
        for sample in review_samples:
            sample.rating = decision_tree.set_label
            return
    left_indices = [i for (i, sample) in enumerate(review_samples)
                    if decision_tree.split_word in sample.word_list]
    left_set = [sample for (i, sample) in enumerate(review_samples) if i in left_indices]
    right_set = [sample for (i, sample) in enumerate(review_samples) if i not in left_indices]
    test(left_set, decision_tree.left)
    test(right_set, decision_tree.right)

