__author__ = 'thomas'

import gensim
import os
import numpy as np
import itertools
import scipy.misc
from scipy import spatial

import pickle

cache_subdir = ""
model_file_name = "lda_model"
cos_sims_file_name = "sims.cos"
hellinger_sims_file_name = "sims.hell"


class LdaCalc:
    def __init__(self, bows=None, id2word=None, lda_cache_dir="", sims_cache_dir="", hell_threshold=0.98, cos_threshold=0.8, num_topics=100):
        self.bows = bows  # {fileid:bowvector}
        self.bow_vector_list = bows.values()
        self.id2word = id2word
        self.cache_dir = sims_cache_dir
        self.model_file_name = os.path.join(lda_cache_dir, model_file_name)
        self.sims_cache_dir = sims_cache_dir
        self.cos_sims_file_name = os.path.join(self.sims_cache_dir, cos_sims_file_name)
        self.hell_sims_file_name = os.path.join(self.sims_cache_dir, hellinger_sims_file_name)
        self.lda_model = None
        self.num_topics = num_topics
        self.cos_sims = list()
        self.hell_sims = list()
        self.cos_sim_threshold = cos_threshold
        self.hell_sim_threshold = hell_threshold

    def run_lda(self):
        # get an lda-compatible dictionary
        dictionary = gensim.corpora.Dictionary.from_corpus(self.bow_vector_list, self.id2word)

        # run the lda
        self.lda_model = gensim.models.ldamodel.LdaModel(corpus=self.bow_vector_list, id2word=dictionary,
                                                         num_topics=self.num_topics, update_every=1,
                                                         chunksize=10000, passes=1)
        print("done running the lda")

    def print_topics(self):
        # print that shit!
        topics = self.lda_model.print_topics(num_topics=self.num_topics, num_words=20)
        for topic in topics:
            print(topic)

    def save(self):
        self.lda_model .save(self.model_file_name)

    def load(self):
        self.lda_model = gensim.models.LdaModel.load(self.model_file_name)
        print("loaded")

    def get_sim_cos(self, vec1, vec2):
        #sim = gensim.matutils.cossim(vec1, vec2) <-- note:  this requires sparse arrays
        sim = 1.0 - spatial.distance.cosine(vec1, vec2)
        return sim

    def get_sim_hellinger(self, vec1, vec2):
        #Hellinger distance is useful for similarity between probability distributions (such as LDA topics):
        sim = 1.0 - np.sqrt(0.5 * ((np.sqrt(vec1) - np.sqrt(vec2))**2).sum())
        return sim

    def get_sim_euclidean(self, vec1, vec2):
        euclidean = spatial.distance.euclidean(vec1, vec2)
        return euclidean

    def get_sims_sorted(self, sims):
        sorted_sims = sorted(sims, key=lambda tup: tup[2], reverse=True)
        return sorted_sims

    def calc_sim(self, topicid1, topicid2):
        if topicid1 == topicid2:
            return

        topics1 = self.lda_model.get_document_topics(self.bows[topicid1])
        topics2 = self.lda_model.get_document_topics(self.bows[topicid2])

        hellsim = self.get_sim_hellinger(topics1, topics2)
        cossim = self.get_sim_cos(topics1, topics2)

        #if self.hell_sim_threshold <= hellsim:
        self.hell_sims.append((topicid1, topicid2, hellsim))
        #if self.cos_sim_threshold <= cossim:
        self.cos_sims.append((topicid1, topicid2, cossim))
        return hellsim, cossim

    def calc_sims_for_topic_distribution(self, topic_distribution, sim_method="Cosine"):
        sims = []
        debug_i = 0
        debug_max = 0  # set to 0 for no debugging
        for topicid in self.bows.keys():
            topics_sparse = self.lda_model.get_document_topics(self.bows[topicid])
            topics_full = gensim.matutils.sparse2full(topics_sparse, self.lda_model.num_topics)

            if sim_method == "Cosine":
                sim = self.get_sim_cos(topic_distribution, topics_full)
                sims.append((topicid, sim))
            elif sim_method == "Euclidean":
                # for euclidean, lower numbers are better
                sim = self.get_sim_euclidean(topic_distribution, topics_full)
                sims.append((topicid, sim))
            else:  # hellinger
                sim = self.get_sim_hellinger(topic_distribution, topics_full)
                sims.append((topicid, sim))
            debug_i += 1
            if debug_max != 0 and debug_i > debug_max:
                break

        if sim_method != "Euclidean":
            sorted_sims = sorted(sims, key=lambda x: x[1], reverse=True)
        else:  # if it's euclidean, the lowest numbers are the best ones.
            sorted_sims = sorted(sims, key=lambda x: x[1], reverse=False)
        return sorted_sims

    def calc_sims(self):
        i = 0

        num_combos = scipy.misc.comb(len(self.bows.keys()), 2)
        print("########################################################################################\n")
        for fileid1, fileid2 in itertools.combinations(self.bows.keys(), 2):
            i += 1
            sims = self.calc_sim(fileid1, fileid2)
            print("(%d of %d) comparing %s and %s: hell=%f, cos=%f" \
                  % (i, num_combos, fileid1, fileid2, sims[0], sims[1]))

        self.hell_sims = self.get_sims_sorted(self.hell_sims)
        self.cos_sims = self.get_sims_sorted(self.cos_sims)

    def save_sims(self):
        pickle.dump(self.cos_sims, open(self.cos_sims_file_name, "wb"))
        pickle.dump(self.hell_sims, open(self.hell_sims_file_name, "wb"))

    def load_sims(self):
        self.cos_sims = pickle.load(open(self.cos_sims_file_name, "rb"))
        self.hell_sims = pickle.load(open(self.hell_sims_file_name, "rb"))

