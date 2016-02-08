__author__ = 'thomas'

from flask import Flask

from flask import request
import os
from corpus.bowBuilder import BowBuilder
from lda.ldaCalc import LdaCalc
import json
from flask.ext.cors import CORS
import gensim
from lda.dynamicLda import DynamicLda

from datetime import timedelta
from flask import make_response, request, current_app
from functools import update_wrapper
from reuters.vocabulary import Vocabulary
from reuters.text import Text

sims_cache_path = "/home/thomas/projects/clms/internship/lda/cache/sims"
reuters_cache_path = "/home/thomas/projects/clms/internship/lda/cache/reuters"
reuters_path = "/home/thomas/Downloads/corpora/reuters/"
vocab_file = os.path.join(reuters_path, "stem.termid.idf.map.txt")
text_file = os.path.join(reuters_path, "lyrl2004_tokens_train.dat")

N = 20   # num of sims to return'
app = Flask(__name__)
CORS(app)
counter = 0
lda_model = None
vocab = None
text_corpus = None
cache_root = ""


@app.route('/')
def health():
    return 'Lda Server is totally working, you guys!'


def build_json_list(topn):
    retstr = '['
    for i in range(0, N):
        if i != 0:
            retstr += ","
        retstr += '{"id":%d, "name":"%s", "sim": %f}' % (i, topn[i][0], topn[i][1])
    retstr += ']'
    return retstr


def read_file(fileid):
    filepath = os.path.join(cache_root, "plaintext", fileid)
    try:
        with open(filepath, "r") as datafile:
            data = datafile.read()
        return data
    except:
        return "Error Reading File: " + filepath


@app.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    #  return read_file(doc_id)
    return " ".join(text_corpus.text_vectors[doc_id])


@app.route('/get-term-sim', methods=['POST'])
def get_term_sim():
    request_data = request.data
    json_data = json.loads(request_data)
    query_terms = json_data["query_terms"]
    sim_method = json_data["sim_method"]
    print "Terms: " + str(query_terms)
    if request.method == 'POST':
        pass
    else:
        pass
    vec_bow = lda_model.lda_model.id2word.doc2bow(query_terms)
    vec_lda = lda_model.lda_model[vec_bow]
    sims = lda_model.calc_sims_for_topic_distribution(vec_lda, sim_method)
    topn = sims[:N]
    json_list = build_json_list(topn)
    # topn_json = json.dumps(topn)
    # print str(topn_json)
    return json_list


def get_id_list(json_array):
    ids = []
    for item in json_array:
        ids.append(item["name"])
    return ids


@app.route('/get-sims-from-concept', methods=['POST'])
def get_sims_from_concept():
    request_data = request.data
    json_data = json.loads(request_data)
    query_terms = json_data["query_terms"]
    sim_method = json_data["sim_method"]
    good_doc_ids = get_id_list(json_data["good_doc_ids"])
    bad_doc_ids = get_id_list(json_data["bad_doc_ids"])
    good_terms = json_data["good_terms"]
    bad_terms = json_data["bad_terms"]

    print "Terms: " + str(query_terms)
    if request.method == 'POST':
        pass
    else:
        pass
    dynamicLda = DynamicLda(lda_model, lda_model.bows)
    biased_topic_distribution = dynamicLda.get_biased_topic_distribution(query_terms, good_doc_ids, bad_doc_ids,
                                                                         good_terms, bad_terms)

    sims = lda_model.calc_sims_for_topic_distribution(biased_topic_distribution, sim_method)
    topn = sims[:N]
    json_list = build_json_list(topn)
    # topn_json = json.dumps(topn)
    # print str(topn_json)
    return json_list

if __name__ == '__main__':
    print "Loading BoW Vectors from cache...\n"
    vocab = Vocabulary(vocab_file)
    vocab.load_from_cache(reuters_cache_path)
    text_corpus = Text(text_file, vocab)
    text_corpus.load_from_cache(reuters_cache_path)

    print "Loading LDA model...\n"
    lda_model = LdaCalc(bows=text_corpus.bow_vectors,
                        sims_cache_dir=sims_cache_path,
                        lda_cache_dir=reuters_cache_path,
                        num_topics=100)
    lda_model.load()
    lda_model.print_topics()

    print "\nStarting Server..."
    app.run(host="0.0.0.0", port=1338, debug=True, use_reloader=False)
