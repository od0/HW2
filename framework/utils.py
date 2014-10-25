INPUT_FILE1 = ('/Users/sean/Sync/cornell/modern-analytics-fall-2014/'
  'hw2-bucket/data/finefoods-small.txt')
INPUT_FILE2 = ('/home/sean/Sync/cornell/modern-analytics-fall-2014/'
               'hw2-bucket/data/finefoods-small.txt')

REVIEW_INDEX = 0
SCORE_INDEX = 1
UNIGRAMS_INDEX = 0
WORD_COUNT_INDEX = 1
LABEL_DESC = ['Bad', 'Good']
TREE_OUTFILE = 'decisiontree.pdf'

# decision tree stuff
def entropy():
    raise 'not implemented'

def information_gain():
    raise 'not implemented'

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
