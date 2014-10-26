from collections import OrderedDict

import scan
import utils
import config
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
    results = []
    for sample in data:
        results.append(ReviewSample(sample))
        #sample_unigrams = utils.get_unigram(sample[config.REVIEW_INDEX])
        #sample_unigrams.append(data[config.SCORE_INDEX])
        #results.append((sample_unigrams, sample[config.SCORE_INDEX]))
    return results
    #return [utils.get_unigram(sample[config.REVIEW_INDEX]) for sample in data]


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


def problem2d(train_data):
    review_samples = generate_unigrams(train_data)
    positive_counts, top_positive = samples_by_label(review_samples, 500, 1)
    negative_counts, top_negative = samples_by_label(review_samples, 500, 0)
    feature
    print
    print 'H(S) = %0.5f' % utils.entropy([sample.rating for sample in review_samples])


def derive_features(positive_counts, negative_counts):
    features = set()
    features = features.union([word for word, count in positive_counts.iteritems()],
                              [word for word, count in negative_counts.iteritems()])

    feature_counts = {}
    for word in features:
        feature_counts[word] = (positive_counts.setdefault(word, 0) +
                                negative_counts.setdefault(word, 0))

    return feature_counts


'''def normalize_data_set(length, data, features, unigram_list):
    #word_count = sum([record[config.WORD_COUNT_INDEX] for record in unigram_list])
    data_set = []
    class_set = []
    for i in range(length):
        sample = unigram_list[i]
        feature_vector = [
            sample[config.UNIGRAMS_INDEX].setdefault(word, 0)
            for word in features
        ]
        data_set.append(feature_vector)
        class_set.append(data[i][config.SCORE_INDEX])

    return data_set, class_set'''


RUN_FILTER = {
    'full': False,
    '2a': True,
    '2b': True,
    '2c': True,
    '2d': True,
    '2e': False,
    '2f': False
}


def main():
    #binary_label = True
    if RUN_FILTER['full']:
        infile = config.INPUT_FILE
    else:
        infile = config.INPUT_FILE_SAMPLE

    data, data_filtered, train_data, test_data = None, None, None, None
    if RUN_FILTER['2a'] or RUN_FILTER['2b']:
        data = scan.scan(infile, exclude_stopwords=False, binary_label=True)
        length = len(data)
    if RUN_FILTER['2c'] or RUN_FILTER['2d'] or RUN_FILTER['2e'] or RUN_FILTER['2f']:
        data_filtered = scan.scan(infile, exclude_stopwords=True, binary_label=True)
        length = len(data_filtered)
        train_data = data_filtered[:int(length*.8)]
        test_data = data_filtered[int(length*.8):]

    #data = scan.scan(infile, exclude_stopwords=False, binary_label=True)

    if RUN_FILTER['2a']:
        problem2a(data, 10)

    if RUN_FILTER['2b']:
        problem2b(data)

    if RUN_FILTER['2c']:
        problem2c(data_filtered)

    if RUN_FILTER['2d']:
        problem2d(train_data)


    #decision_tree = dt.train(train_data)
    #test_results = dt.test(decision_tree, test_data)

    #print test_results

if __name__ == '__main__':
    main()
