import config
import utils


class ReviewSample(object):
    def __init__(self, datum):
        unigrams = utils.get_unigram(datum[config.REVIEW_INDEX])
        self.unigrams = unigrams[config.UNIGRAMS_INDEX]
        self.word_count = unigrams[config.WORD_COUNT_INDEX]
        self.rating = datum[config.SCORE_INDEX]
        self.predicted_rating = None


    @property
    def word_list(self):
        return self.unigrams.keys()

    def __repr__(self):
        return ("<ReviewSample(rating=%d, word_count=%d, unigrams=%s)>" %
                (self.rating, self.word_count, self.unigrams))