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

    def _get_label(self, review_set):
        max_label = max([sample.rating for sample in review_set])
        if not 1 >= max_label >= 0:
            raise AttributeError("Ivalid label: %d. Decision tree label must be 0 or 1." % max_label)
        self.node_label = max_label

    def decision(self, review_set, features, depth):
        entropy = utils.entropy([sample.rating for sample in review_set])
        if len(review_set) == 2033:
            print

        if 1 > entropy > 0:
            full_set_length = len(review_set)
            # max_info_gain holds info gain
            max_info_gain = (0, '', None, None)
            for word in features:
                left_labels, right_labels, left_indices = DecisionTree.split(word, review_set)
                info_gain = utils.information_gain(entropy, full_set_length, left_labels, right_labels)
                if info_gain > max_info_gain[0]:
                    left_set = [sample for (i, sample) in enumerate(review_set) if i in left_indices]
                    right_set = [sample for (i, sample) in enumerate(review_set) if i not in left_indices]
                    max_info_gain = (info_gain, word, left_set, right_set)
                #info_gain_features[word] = (info_gain, left_set, right_set)

            # Exclude this word/attribute
            try:
                self.split_word = max_info_gain[1]
                features.remove(self.split_word)
                print '\tSplitting on %s (info gain %0.5f, branch size %d)' % (
                    self.split_word, max_info_gain[0], len(review_set)
                )
                if max_info_gain[2]:
                    self.left = DecisionTree(max_info_gain[2], features,
                                             depth=depth+1, desc='%s appears' % self.split_word)
                if max_info_gain[3]:
                    self.right = DecisionTree(max_info_gain[3], features,
                                              depth=depth+1, desc='%s does not appear' % self.split_word)
            except KeyError as e:
                print "\t\tError: %s\n\t\tEntropy: %0.10f\n\t\tReviews: %d\n\t\tFeatures: %d\n\t\tMax Info:%s" % (
                    e, entropy, len(review_set), len(features), max_info_gain
                )
        else:
            self._get_label(review_set)

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
    raise NotImplementedError

