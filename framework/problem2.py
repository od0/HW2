import scan
import utils


def problem2a(data):
    sample = data.pop()
    label = scan.score_to_binary(sample[utils.SCORE_INDEX])
    print '2a) Example review and label pair from sample data: '
    print ('\tReview: %s' % sample[utils.REVIEW_INDEX])
    print ('\tLabel: %s (%s)' % (label, utils.LABEL_DESC[label]))


def problem2b(data):
    unigrams_list = [
        utils.get_unigram(sample[utils.REVIEW_INDEX])
        for sample in data
    ]

    positive_counts, top_positive = aggregate_unigrams(
        [bucket for i, bucket in enumerate(unigrams_list)
         if scan.score_to_binary(data[i][utils.SCORE_INDEX])],
        30)

    negative_counts, top_negative = aggregate_unigrams(
        [bucket for i, bucket in enumerate(unigrams_list)
         if not scan.score_to_binary(data[i][utils.SCORE_INDEX])],
        30)

    print 'Top words in positive reviews:'
    print_top_words(top_positive)
    print
    print 'Top words in negative reviews:'
    print_top_words(top_negative)


def aggregate_unigrams(unigram_list, top_num):
    word_counts = {}
    #top_words = [{str(i): 0} for i in range(top_num)]
    top_words = [('', 0) for i in range(top_num)]
    min_count = 0
    for record in unigram_list:
        for word, count in record[utils.UNIGRAMS_INDEX].iteritems():
            word_counts[word] = word_counts.setdefault(word, 0) + count
            if word_counts[word] > min_count:
                min_count = update_top_words(word_counts, word, top_words)
    return word_counts, top_words


def update_top_words(word_counts, word, top_words):
    # Check whether word exists in top_words already
    match_index = [i for (i, top) in enumerate(top_words) if word in top]

    # Update the count in top words if it does appear in top_words
    new_top = (word, word_counts[word])
    if match_index:
        top_words[match_index.pop()] = new_top
    else:
        top_words.append(new_top)
        top_words.sort(key=(lambda x: x[1]), reverse=True)
        top_words.pop()
    return top_words[-1][1]


def print_top_words(top_words):
    for i, result in enumerate(top_words):
        print('\t#%3d: %s (%dx)' % (i+1, result[0], result[1]))


def main():
    binary_label = True

    data = scan.scan(utils.INPUT_FILE2, binary_label)
    length = len(data)

    train_data = data[:int(length*.8)]
    test_data = data[int(length*.8):]

    problem2a(data)
    problem2b(data)

    #decision_tree = dt.train(train_data)
    #test_results = dt.test(decision_tree, test_data)

    #print test_results

if __name__ == '__main__':
    main()
