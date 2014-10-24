import re, string
import nltk
from nltk.corpus import stopwords

# constants
LINES_PER_REVIEW = 8
BINARY = True
NONWORDS = re.compile('[\W_]+')
STOPWORDS = stopwords.words('english')

# read in a file
def scan(filename, exclude_stopwords = False, binary_label = False):
    data = []
    with open(filename, 'r') as f:
        elements = []
        for i in range(LINES_PER_REVIEW):
            elements.append(f.readline().split(':', 1)[1])

        review = (elements[6] + ' ' + elements[7])
        review = ' '.join(re.split(NONWORDS, review))
        review = review.strip().lower()

        if exclude_stopwords:
            review = ' '.join([w for w in review.split() if w not in STOPWORDS])

        score = float(elements[4].strip())

        if binary_label:
            score = score_to_binary(score)

        datum = [review, score]
        data.append(datum)
    return data

def score_to_binary(score):
    if score >= 4:
        return 1
    else:
        return 0
