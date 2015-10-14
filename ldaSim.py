__author__ = 'thomas'

import os
from corpus.bowBuilder import BowBuilder

from lda.ldaCalc import LdaCalc

cache_root = os.path.join(os.getcwd(), "cache")

bowBuilder = BowBuilder(cache_dir=cache_root)
bowBuilder.load()
bows = bowBuilder.bowVectorCorpus

print "\nLoading LDA model -----------------------------------------------\n"
lda = LdaCalc(bows=bows, cache_dir=cache_root)
lda.load()
lda.print_topics()

lda.calc_sims()
lda.save_sims()



