import config
import utils
import scan


class ReviewSample(object):
    def __init__(self, datum):
        unigrams = utils.get_unigram(datum[config.REVIEW_INDEX])
        self.unigrams = unigrams[config.UNIGRAMS_INDEX]
        self.word_count = unigrams[config.WORD_COUNT_INDEX]
        self.rating = datum[config.SCORE_INDEX]

    @property
    def words(self):
        return self.unigrams.keys()