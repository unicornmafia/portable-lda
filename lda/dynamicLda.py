__author__ = 'thomas'

import numpy as np
import gensim


class DynamicLda:
    def __init__(self, lda, bows):
        self.lda = lda
        self.topics = lda.lda_model.state.get_lambda()
        self.lda_model = lda.lda_model
        self.bows = bows
        self.topic_threshold = 1.0/self.lda_model.num_topics

    def get_topic_distributions(self, ids):
        distributions = np.ndarray(shape=(len(ids), len(self.topics)), dtype=float)
        i = 0
        for id in ids:
            bow = self.bows[id]
            topics = self.lda_model.get_document_topics(bow)
            full_topics = gensim.matutils.sparse2full(topics, self.lda_model.num_topics)
            full_topics[full_topics <= self.topic_threshold] = 0.0
            full_topics[full_topics > self.topic_threshold] = 1.0
            distributions[i] = full_topics
            i += 1
        if i == 0:
            distributions = np.array([np.zeros(len(self.topics))])
        return distributions

    def get_topic_bias(self, good_docs, bad_docs):
        good_dist = self.get_topic_distributions(good_docs)
        summed_good_dist = good_dist.sum(axis=0)
        bad_dist = self.get_topic_distributions(bad_docs)*(-1)
        summed_bad_dist = bad_dist.sum(axis=0)
        summed_dist = summed_good_dist + summed_bad_dist
        summed_dist[summed_dist > 0] = self.topic_threshold
        summed_dist[summed_dist < 0] = -self.topic_threshold
        return summed_dist

    def get_biased_topic_distribution(self, query_terms, good_docs, bad_docs, good_terms, bad_terms):
        # get the bias based on concept
        bias = self.get_topic_bias(good_docs, bad_docs)

        # make the query terms into a bow and get the topic distribution from the model for that bow
        vec_bow = self.lda_model.id2word.doc2bow(query_terms)
        vec_topic_distribution = gensim.matutils.sparse2full(self.lda_model[vec_bow], self.lda_model.num_topics)

        # add bias to topic distribution
        biased_topic_distribution = vec_topic_distribution + bias

        # remove negative values and renormalize
        biased_topic_distribution[biased_topic_distribution < 0] = 0
        biased_topic_distribution /= sum(biased_topic_distribution)
        return biased_topic_distribution



