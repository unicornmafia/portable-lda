__author__ = 'thomas'
import os
import gensim
from reuters.vocabulary import Vocabulary
from reuters.text import Text
from reuters.vectors import Vectors

reuters_cache_path = "/home/thomas/projects/clms/internship/lda/cache/reuters"
reuters_path = "/home/thomas/Downloads/corpora/reuters/"
vocab_file = os.path.join(reuters_path, "stem.termid.idf.map.txt")
text_file = os.path.join(reuters_path, "lyrl2004_tokens_train.dat")
vector_file = os.path.join(reuters_path, "lyrl2004_vectors_train.dat")

print("Loading Vocabulary...")
vocab = Vocabulary(vocab_file)
vocab.load_from_text()
print("Saving Vocabulary to Cache...")
vocab.save_to_cache(reuters_cache_path)
#vectors = Vectors(vector_file, 100)
#vectors.load_from_text()
print("Loading Corpus...")
text = Text(text_file, vocab)
text.load_from_text()
print("Saving Corpus to Cache...")
text.save_to_cache(reuters_cache_path)

num_topics = 30

print("Making Dictionary from Corpus (" + str(len(text.bow_vectors)) + " documents)...")
# get an lda-compatible dictionary
dictionary = gensim.corpora.Dictionary.from_corpus(text.bow_vectors.values(), vocab.id2word)

print("Training Model...")
# run the lda
lda_model = gensim.models.ldamodel.LdaModel(corpus=text.bow_vectors.values(), id2word=dictionary,
                                                         num_topics=num_topics, update_every=1,
                                                         chunksize=10000, passes=1)

print("Saving Model...")
lda_model.save(os.path.join(reuters_cache_path, "lda_model"))

