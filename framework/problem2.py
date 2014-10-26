from __future__ import division
from collections import OrderedDict
import logging
logging.basicConfig(filename='logs/decision.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
import utils
import scan
import config
import decision_tree
from unigrams import ReviewSample


def problem2a(data, num):
    print '2a) %d Example review and label pairs from data: ' % num
    for i in range(num):
        sample = data.pop()
        label = sample[config.SCORE_INDEX]
        print ('\tReview: %s' % sample[config.REVIEW_INDEX])
        print ('\tLabel: %s (%s)' % (label, config.LABEL_DESC[label]))


def samples_by_label(review_samples, top_num, label=1):
    return aggregate_unigrams(
        [sample for i, sample in enumerate(review_samples) if sample.rating == label],
        top_num)


def report_top_n_words(review_samples, n):
    positive_counts, top_positive = samples_by_label(review_samples, n, 1)
    negative_counts, top_negative = samples_by_label(review_samples, n, 0)

    print '\tPositive reviews:'
    print_top_words(top_positive)
    print
    print '\tNegative reviews:'
    print_top_words(top_negative)


def generate_unigrams(data):
    return [ReviewSample(sample) for sample in data]


def aggregate_unigrams(all_samples, top_num):
    word_counts = {}
    for sample in all_samples:
        for word, count in sample.unigrams.iteritems():
            word_counts[word] = word_counts.setdefault(word, 0) + count
    top_words = get_top_n_words_counts(top_num, word_counts)
    return word_counts, top_words


def get_top_n_words_counts(num, word_counts):
    result = sorted(word_counts.items(), key=lambda t: t[1], reverse=True)
    return OrderedDict(result[:num])


def print_top_words(top_words):
    for i, result in enumerate(top_words.items()):
        print('\t#%3d: %s (%dx)' % (i+1, result[0], result[1]))


def problem2b(data):
    print
    print 'Top words from %d sample reviews' % (len(data))
    review_samples = generate_unigrams(data)
    report_top_n_words(review_samples, 30)


def problem2c(data):
    print
    print
    review_samples = generate_unigrams(data)
    print 'Top words (excluding stopwords) from %d sample reviews' % (len(data))
    report_top_n_words(review_samples, 30)


def problem2e(train_data, test_data):
    print 'Beginning decision tree training using ID3 algorithm'
    review_samples = generate_unigrams(train_data)
    positive_counts, top_positive = samples_by_label(review_samples, 500, 1)
    negative_counts, top_negative = samples_by_label(review_samples, 500, 0)

    feature_set = derive_features(top_positive, top_negative)
    print ('\tDecision tree feature/attribute set includes %d total words' % len(feature_set))
    print ('\tStarting entropy of the review set with %d samples is %0.5f' %
           (len(review_samples), utils.entropy([sample.rating for sample in review_samples])))

    return decision_tree.train(review_samples, feature_set)


def derive_features(positive_counts, negative_counts):
    features = set()
    features = features.union([word for word, count in positive_counts.iteritems()],
                              [word for word, count in negative_counts.iteritems()])
    return features


def problem2f(test_data, d_tree):
    print 'Predicting scores for test %d review samples with decision tree' % len(test_data)
    review_samples = generate_unigrams(test_data)
    decision_tree.test(review_samples, d_tree)
    total = len(test_data)
    true_positive = len([sample for sample in review_samples
                         if sample.rating == 1 and sample.predicted_rating == 1])
    false_positive = len([sample for sample in review_samples
                         if sample.rating == 0 and sample.predicted_rating == 1])
    true_negative = len([sample for sample in review_samples
                         if sample.rating == 0 and sample.predicted_rating == 0])
    false_negative = len([sample for sample in review_samples
                         if sample.rating == 1 and sample.predicted_rating == 0])
    print 'Finished testing samples'
    print '%0.5f Oveall accuracy (%d/%d)' % (
        (true_positive + true_negative) / total, true_positive + true_negative, total
    )
    print '\t%0.5f True Positive Rate (Sensitivity)' % (true_positive / (true_positive + false_negative))
    print '\t%0.5f True Negative Rate (Specificity)' % (true_negative / (false_positive + true_negative))
    print '\t%0.5f Positive Predictive Value (Precision)' % (true_positive / (true_positive + false_positive))
    print '\t%0.5f False Positive Rate (Fall-Out)' % (false_positive / (false_positive + true_negative))
    print '\t%0.5f False Negative Rate' % (false_negative / (true_positive + false_negative))


def main():
    if config.RUN_FILTER['full']:
        infile = config.INPUT_FILE
    else:
        infile = config.INPUT_FILE_SAMPLE

    data = None
    try:
        if config.RUN_FILTER['2a'] or config.RUN_FILTER['2b']:
            print 'Loading data from file'
            data = scan.scan(infile, exclude_stopwords=False, binary_label=True)
    except IOError as e:
        print 'ERROR: Cannot open file %s. Please provide a valid INPUT_FILE in config.py\n%s' % (infile, e)
        exit(1)

    if config.RUN_FILTER['2a']:
        problem2a(data, 10)

    if config.RUN_FILTER['2b']:
        problem2b(data)

    data_filtered, train_data, test_data = None, None, None
    try:
        if config.RUN_FILTER['2c'] or config.RUN_FILTER['2e'] or config.RUN_FILTER['2f']:
            print 'Loading data from file (excluding stopwords)'
            data_filtered = scan.scan(infile, exclude_stopwords=True, binary_label=True)
            length = len(data_filtered)
            train_data = data_filtered[:int(length*.8)]
            test_data = data_filtered[int(length*.8):]
    except IOError as e:
        print 'ERROR: Cannot open file %s. Please provide a valid INPUT_FILE in config.py\n%s' % (infile, e)
        exit(1)

    if config.RUN_FILTER['2c']:
        problem2c(data_filtered)

    if config.RUN_FILTER['2e'] or config.RUN_FILTER['2f']:
        d_tree = problem2e(train_data, test_data)
        print '\tFinished training decision tree'
        if config.RUN_FILTER['2f']:
            problem2f(test_data, d_tree)

if __name__ == '__main__':
    main()
