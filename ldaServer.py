__author__ = 'thomas'

from flask import Flask

import codecs
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
from reuters.reuters_indexer import ReutersIndex
from reuters.reuters_article_parser import ReutersArticle


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
num_topics = 100
reuters_index = ReutersIndex()
reuters_index.load_dict()
utf8_reader = codecs.getreader("utf-8")


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
    file_path = reuters_index.index[int(fileid)]

    try:
        reuters_article = ReutersArticle(file_path)
        file_text = reuters_article.get_all_text()
        return file_text
    except FileNotFoundError:
        return "Error Reading File: " + file_path


@app.route('/document/<doc_id>', methods=['GET'])
def get_document(doc_id):
    #  return read_file(doc_id)
    return read_file(doc_id)


@app.route('/get-term-sim', methods=['POST'])
def get_term_sim():
    request_data = request.data.decode("utf-8")
    json_data = json.loads(request_data)
    query_terms = json_data["query_terms"]
    sim_method = json_data["sim_method"]
    print("Terms: " + str(query_terms))
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
    request_data = request.data.decode("utf-8")
    json_data = json.loads(request_data)
    query_terms = json_data["query_terms"]
    sim_method = json_data["sim_method"]
    good_doc_ids = get_id_list(json_data["good_doc_ids"])
    bad_doc_ids = get_id_list(json_data["bad_doc_ids"])
    good_terms = json_data["good_terms"]
    bad_terms = json_data["bad_terms"]

    print("Terms: " + str(query_terms))
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
    print("Loading BoW Vectors from cache...\n")
    vocab = Vocabulary(vocab_file)
    vocab.load_from_cache(reuters_cache_path)
    text_corpus = Text(text_file, vocab)
    text_corpus.load_from_cache(reuters_cache_path)

    print("Loading LDA model...\n")
    lda_model = LdaCalc(bows=text_corpus.bow_vectors,
                        sims_cache_dir=sims_cache_path,
                        lda_cache_dir=reuters_cache_path,
                        num_topics=num_topics)
    lda_model.load()
    lda_model.print_topics()

    print("Indexing Model for similarities...")
    index = gensim.similarities.MatrixSimilarity(lda_model.lda_model[text_corpus.bow_vectors])
    doc = "Fear of contracting AIDS from women is prompting some men to turn to children for sex, the head of the United Nations' global AIDS agency told a conference on child sex abuse on Wednesday. \"The AIDS epidemic has become both a cause and a consequence of the trade in children,\" Peter Piot, executive director of UNAIDS, said in a speech. \"Men are looking out for younger girls because they are concerned that if they have sex with adult women then they are at risk for HIV infection,\" Piot told Reuters. Sex with younger partners as protection from HIV, the virus that causes AIDS, is an illusion, Piot told delegates from more than 100 countries on the second day of the first World Congress Against Commercial Sexual Exploitation of Children. Many child prostitutes were infected and young people are actually more susceptible to infection than adults, Piot said. \"Because of the physical disproportion between the partners, a child who is not fully grown is more easily torn or damaged by penetrative sex, and this makes it easier for the virus to pass into the child's body,\" Piot said in a speech at the conference. \"And a child can't fight back, no matter how rough the sex or how long it lasts,\" Piot added. Over 1,000 delegates have gathered in Stockholm for the five-day conference to discuss the scope of the problems, legal reform, and raising public awareness. More than one million children worldwide are reportedly forced into child prostitution, trafficked and sold for sexual purposes and used in the production of child pornography, according to UNICEF figures. About one million children are currently HIV positive or have AIDS. Most contracted the disease from their infected mothers, Piot said. Over two million children had already died from the disease, he said. Statistics showing the rate of HIV infection among child prostitutes were unavailable, but Piot said that very small samples indicated that as many as 50 percent of underage sex workers could have the virus. The conference is jointly organised by the Swedish government, the United Nations Childrens Fund (UNICEF), pressure group ECPAT (End Child Prostitution in Asia Tourism) and the NGO group on the rights of the child. While promoting condom use could curb the spread of the HIV virus among underage sex workers, Piot called for broader, urgent measures from governments and communities to end the sexual trade in children. \"Children are weak, vulnerable and uninformed, and they are scarcely in a position to demand that the client should use a condom,\" Piot said. \"Through income-generation, promotion of rural industry and education policies, governments can reinforce families' resistance to the lure of commercial gain through the sale of their children,\" Piot said as one example. Since the start of the conference, as if to underline how widespread the issue is, horrifying abuse and paedophile cases have come to light in Albania, Australia, Belgium and Finland. Finnish police said on Wednesday they had discovered in a Helsinki flat a massive computer library of exceptionally severe child pornography including pictures of mutilated people and cannibalism. Police had taken two computers and nearly 350 floppy disks from the home of a 19-year-old student, but could not arrest him because they do not have the powers under Finnish law. In Belgium police on Wednesday were digging for human remains at a property owned by Marc Dutroux, chief suspect in a child sex and kidnapping ring. Dutroux had already led police to the bodies of eight-year-olds Melissa Russo and Julie Lejeune and of Weinstein 10 days ago. They were buried in the garden of one of Dutroux' five other houses in and around the city of Charleroi. Also this week, a 75-year-old Australian man appeared in court charged with 850 child sex crimes, including indecent dealing, sodomy and permitting sodomy with children."
    vec_bow = lda_model.id2word.doc2bow(doc.lower().split())
    vec_lda = lda_model.lda_model[vec_bow]
    sims = index[vec_lda]
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sims)

    print("\nStarting Server...")
    app.run(host="0.0.0.0", port=1338, debug=True, use_reloader=False, threaded=True)
