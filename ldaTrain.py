__author__ = 'thomas'

import os

from corpus.textExtractor import TextExtractor
from corpus.bowBuilder import BowBuilder
from lda.ldaCalc import LdaCalc

cache_root = os.path.join(os.getcwd(), "cache")

# start here!
# get text
print("\nextracting text from corpus ------------------------------------\n")
extractor = TextExtractor(cache_root)
extractor.get_texts()
# extractor.save()

# convert to BOW vectors
print("\nbuilding BOW vectors from corpus ------------------------------------\n")
bowBuilder = BowBuilder(docs=extractor.texts, cache_dir=cache_root)
bowBuilder.generate_bows()
bowBuilder.save()

# run the LDA
print("\ntraining LDA model -----------------------------------------------\n")
lda = LdaCalc(bowBuilder.bowVectorCorpus, bowBuilder.id2word, cache_root)
lda.run_lda()
print("\nsaving LDA model -----------------------------------------------\n")
lda.save()
lda.print_topics()


