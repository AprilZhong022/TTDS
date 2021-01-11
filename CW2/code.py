import random
import re
from stemming.porter2 import stem
import math
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from collections import Counter
import string
import scipy
from scipy import sparse
import sklearn
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

# evaluate each system
def Eval(system_results, qrels):
    # initialize ir_eval dataframe
    ir_eval = pd.DataFrame(columns=["system_number","query_number","P@10","R@50","r-precision","AP","nDCG@10","nDCG@20"])
    # loop through each system
    for sys_num in range(1,7):
        # read number of system_results for each query
        qr_dic = dict(system_results[system_results["system_number"] == sys_num].query_number.value_counts())
        # read system_results of each system
        sub_results = system_results[system_results["system_number"] == sys_num]
        # initialize
        ireval = pd.DataFrame(columns=["system_number","query_number","P@10","R@50","r-precision","AP","nDCG@10","nDCG@20"])
        ireval["query_number"] = np.array(range(1,11))
        ireval["system_number"] = np.ones((10,1))*(sys_num)
        P_10 = []
        R_50 = []
        r_precision = []
        AP = []
        nDCG_10 = []
        nDCG_20 = []
        index = 0
        # loop through each query
        for qr_id in range(1,11):
            qrel = list(qrels[qrels["query_id"] == qr_id]["doc_id"])
            # calculate P@10
            result1 = list(sub_results[index: index + 10]["doc_number"])
            P_10.append(len(set(result1) & set(qrel)) / 10)
            # calculate R@50
            result2 = list(sub_results[index: index + 50]["doc_number"])
            R_50.append(len(set(result2) & set(qrel)) / len(qrel))
            # calculate r_precision
            result3 = list(sub_results[index: index + len(qrel)]["doc_number"])
            r_precision.append(len(set(result3) & set(qrel)) / len(qrel))
            # calculate AP
            result4 = sub_results[index: index + qr_dic[qr_id]]
            ap_list = []
            for doc in qrel:
                if doc in list(result4["doc_number"]):
                    ap_list.append(int(result4[result4["doc_number"] == doc]["rank_of_doc"]))
            count = 1
            total = 0
            for idx in sorted(ap_list):
                total += count / idx
                count += 1
            AP.append(total/len(qrel))
            # calculate nDCG@10
            result5 = sub_results[index: index + 10]
            # calculate DCG
            DCG10 = 0
            for doc in qrel:
                if doc in list(result5["doc_number"]):
                    G = int(qrels.loc[(qrels["query_id"] == qr_id) & (qrels["doc_id"] == doc)]["relevance"])
                    if int(result5[result5["doc_number"] == doc]["rank_of_doc"]) == 1:
                        DCG10 += G
                    else:
                        DCG10 += G / math.log2(int(result5[result5["doc_number"] == doc]["rank_of_doc"]))
            rel_list = sorted(list(qrels[qrels["query_id"] == qr_id]["relevance"]),reverse=True)
            # calculate iDCG
            iDCG_10 = rel_list[0]
            if len(rel_list) <= 10:
                for i in range(1,len(rel_list)):
                    iDCG_10 += rel_list[i] / math.log2(i+1)
            else:
                for i in range(1,10):
                    iDCG_10 += rel_list[i] / math.log2(i+1)
            nDCG_10.append(DCG10 / iDCG_10)
            # calculate nDCG@20
            result6 = sub_results[index: index + 20]
            # calculate DCG
            DCG20 = 0
            for doc in qrel:
                if doc in list(result6["doc_number"]):
                    G = int(qrels.loc[(qrels["query_id"] == qr_id) & (qrels["doc_id"] == doc)]["relevance"])
                    if int(result6[result6["doc_number"] == doc]["rank_of_doc"]) == 1:
                        DCG20 += G
                    else:
                        DCG20 += G / math.log2(int(result6[result6["doc_number"] == doc]["rank_of_doc"]))
            # calculate iDCG
            rel_list = sorted(list(qrels[qrels["query_id"] == qr_id]["relevance"]),reverse=True)
            iDCG_20 = rel_list[0]
            if len(rel_list) <= 20:
                for i in range(1,len(rel_list)):
                    iDCG_20 += rel_list[i] / math.log2(i+1)
            else:
                for i in range(1,20):
                    iDCG_20 += rel_list[i] / math.log2(i+1)
            nDCG_20.append(DCG20/iDCG_20)
            #renew index
            index += qr_dic[qr_id]
        # assign results to dataframe
        ireval["P@10"] = np.array(P_10)
        ireval["R@50"] = np.array(R_50)
        ireval["r-precision"] = np.array(r_precision)
        ireval["AP"] = np.array(AP)
        ireval["nDCG@10"] = np.array(nDCG_10)
        ireval["nDCG@20"] = np.array(nDCG_20)
        ireval.index = ireval.index + 1
        ireval.loc["mean"] = ireval.mean()
        # concat dataframe
        ir_eval = pd.concat([ir_eval,ireval], axis=0)
    # return ir_eval dataframe
    return ir_eval


# preprocess data and divide into three corpora
def Preprocess_and_divide(data, stopwords_list):
    lines = data.split("\n")
    corpus_Quran = []
    corpus_NT = []
    corpus_OT = []
    Quran_dic = {}
    NT_dic = {}
    OT_dic = {}
    for line in lines:
        # preprocessing
        corpus_name, text = line.split("\t")
        content_withoutoken = re.sub(r"[^\w]+", " ", text).lower()
        words_list = content_withoutoken.split(" ")
        words = filter(lambda x: x not in stopwords_list and x != "", words_list)
        terms = [stem(word) for word in words]
        # divide into three corpora and compute number of documents each term appears
        if corpus_name == "Quran":
            corpus_Quran.append(terms)
            for term in set(terms):
                if term not in Quran_dic:
                    Quran_dic[term] = 1
                else:
                    Quran_dic[term] += 1
        elif corpus_name == "NT":
            corpus_NT.append(terms)
            for term in set(terms):
                if term not in NT_dic:
                    NT_dic[term] = 1
                else:
                    NT_dic[term] += 1
        else:
            corpus_OT.append(terms)
            for term in set(terms):
                if term not in OT_dic:
                    OT_dic[term] = 1
                else:
                    OT_dic[term] += 1
    return corpus_Quran, corpus_NT, corpus_OT, Quran_dic, NT_dic, OT_dic


# compute MI and Chi_square score
def Corpus_scores(total_dic, corpus_dic, corpus_Quran, corpus_NT, corpus_OT):
    MI_dic = {}
    Chi_square_dic = {}
    # divide into one corpus and the other
    if corpus_dic == Quran_dic:
        X, Y = Counter(OT_dic), Counter(NT_dic)
        other_dic = dict(X + Y)
        total_1 = len(corpus_Quran)
        total_2 = len(corpus_NT) + len(corpus_OT)
    elif corpus_dic == OT_dic:
        X, Y = Counter(Quran_dic), Counter(NT_dic)
        other_dic = dict(X + Y)
        total_1 = len(corpus_OT)
        total_2 = len(corpus_NT) + len(corpus_Quran)
    else:
        X, Y = Counter(Quran_dic), Counter(OT_dic)
        other_dic = dict(X + Y)
        total_1 = len(corpus_NT)
        total_2 = len(corpus_Quran) + len(corpus_OT)
    # total number of terms
    N = total_1 + total_2
    for term in total_dic.keys():
        if term in corpus_dic.keys():
            # compute MI score
            N11 = corpus_dic[term]
            N01 = total_1 - N11
            if term in other_dic.keys():
                N10 = other_dic[term]
                N00 = total_2 - N10
                MI = N11 / N * math.log2(float(N * N11) / float((N10 + N11) * (N01 + N11))) \
                     + N01 / N * math.log2(float(N * N01) / float((N00 + N01) * (N01 + N11))) \
                     + N10 / N * math.log2(float(N * N10) / float((N10 + N11) * (N00 + N10))) \
                     + N00 / N * math.log2(float(N * N00) / float((N00 + N01) * (N00 + N10)))
            else:
                N10 = 0
                N00 = total_2
                MI = N11 / N * math.log2(float(N * N11) / float((N10 + N11) * (N01 + N11))) \
                     + N01 / N * math.log2(float(N * N01) / float((N00 + N01) * (N01 + N11))) \
                     + N00 / N * math.log2(float(N * N00) / float((N00 + N01) * (N00 + N10)))
        else:
            N11 = 0
            N01 = total_1
            N10 = other_dic[term]
            N00 = total_2 - N10
            MI = N01/N*math.log2(float(N*N01) / float((N00+N01)*(N01+N11))) \
                 +N10/N*math.log2(float(N*N10) / float((N10+N11)*(N00+N10))) \
                 +N00/N*math.log2(float(N*N00) / float((N00+N01)*(N00+N10)))

        MI_dic[term] = MI
        # compute Chi_square score
        Chi_square = ((N11 + N10 + N01 + N00) * math.pow(N11 * N00 - N10 * N01, 2)) / (
                    (N11 + N01) * (N11 + N10) * (N10 + N00) * (N01 + N00))
        Chi_square_dic[term] = Chi_square
    return MI_dic, Chi_square_dic


# run LDA model
def LDA_model(corpus_Quran, corpus_NT, corpus_OT):
    # run LDA on the entire set of verses from all corpora
    total_corpus = corpus_Quran + corpus_NT + corpus_OT
    dictionary = Dictionary(total_corpus)
    dictionary.filter_extremes(no_below=50, no_above=0.1)
    corpus = [dictionary.doc2bow(text) for text in total_corpus]
    lda = LdaModel(corpus, num_topics=20, id2word=dictionary, random_state=1)
    # compute document-topic probability for Quran
    dictionary1 = Dictionary(corpus_Quran)
    dictionary1.filter_extremes(no_below=50, no_above=0.1)
    corpus1 = [dictionary1.doc2bow(text) for text in corpus_Quran]
    topics_Quran = lda.get_document_topics(corpus1)
    topic_dic_Quran = {}
    for doc in topics_Quran:
        for topic in doc:
            if topic[0] not in topic_dic_Quran.keys():
                topic_dic_Quran[topic[0]] = topic[1]
            else:
                topic_dic_Quran[topic[0]] += topic[1]
    # compute document-topic probability for OT
    dictionary2 = Dictionary(corpus_OT)
    dictionary2.filter_extremes(no_below=50, no_above=0.1)
    corpus2 = [dictionary2.doc2bow(text) for text in corpus_OT]
    topics_OT = lda.get_document_topics(corpus2)
    topic_dic_OT = {}
    for doc in topics_OT:
        for topic in doc:
            if topic[0] not in topic_dic_OT.keys():
                topic_dic_OT[topic[0]] = topic[1]
            else:
                topic_dic_OT[topic[0]] += topic[1]
    # compute document-topic probability for NT
    dictionary3 = Dictionary(corpus_NT)
    dictionary3.filter_extremes(no_below=50, no_above=0.1)
    corpus3 = [dictionary3.doc2bow(text) for text in corpus_NT]
    topics_NT = lda.get_document_topics(corpus3)
    topic_dic_NT = {}
    for doc in topics_NT:
        for topic in doc:
            if topic[0] not in topic_dic_NT.keys():
                topic_dic_NT[topic[0]] = topic[1]
            else:
                topic_dic_NT[topic[0]] += topic[1]
    for k, v in topic_dic_Quran.items():
        topic_dic_Quran[k] = v / len(corpus_Quran)
    for k, v in topic_dic_OT.items():
        topic_dic_OT[k] = v / len(corpus_OT)
    for k, v in topic_dic_NT.items():
        topic_dic_NT[k] = v / len(corpus_NT)
    return lda, topic_dic_Quran, topic_dic_NT, topic_dic_OT


# preprocess data for SVM
def preprocess_data(data):
    chars_to_remove = re.compile(f'[{string.punctuation}]')
    documents = []
    categories = []
    vocab = set([])
    lines = data.split("\n")
    for line in lines:
        line = line.strip()
        if line:
            category, text = line.split("\t")
            # punctuation removal and casefold
            words = chars_to_remove.sub("", text).lower().split()
            for word in words:
                vocab.add(word)
            documents.append(words)
            categories.append(category)
    return documents, categories, vocab


# split train and development data
def Train_Dev_split(preprocessed_data, categories):
    preprocessed_training_data = []
    training_categories = []
    preprocessed_dev_data = []
    dev_categories = []
    random.seed(1)
    # generate random indexes for development set
    dev_index = [random.randint(0, len(preprocessed_data)) for _ in range(len(preprocessed_data) // 10)]
    # get verses of these index for development set
    for i in dev_index:
        preprocessed_dev_data.append(preprocessed_data[i])
        dev_categories.append(categories[i])
    # get verses of other indexes for train set
    train_index = [i for i in range(len(preprocessed_data)) if i not in dev_index]
    for i in train_index:
        preprocessed_training_data.append(preprocessed_data[i])
        training_categories.append(categories[i])
    return preprocessed_training_data, training_categories, preprocessed_dev_data, dev_categories, dev_index


# build a BOW representation of the files
def convert_to_bow_matrix(preprocessed_data, word2id):
    matrix_size = (len(preprocessed_data), len(word2id) + 1)
    # out of vocabulary index
    oov_index = len(word2id)
    # matrix indexed by [doc_id, token_id]
    X = scipy.sparse.dok_matrix(matrix_size)
    for doc_id, doc in enumerate(preprocessed_data):
        for word in doc:
            # add count for the word
            X[doc_id, word2id.get(word, oov_index)] += 1
    return X

# check the model accuracy by precision, recall and f1 score
def accuracy_check(y_true, y_pred, category, cat2id):
    # true positive
    TP = np.sum(np.logical_and(np.equal(y_true, cat2id[category]), np.equal(y_pred, cat2id[category])))
    # false positive
    FP = np.sum(np.logical_and(np.not_equal(y_true, cat2id[category]), np.equal(y_pred, cat2id[category])))
    # true negative
    TN = np.sum(np.logical_and(np.not_equal(y_true, cat2id[category]), np.not_equal(y_pred, cat2id[category])))
    # false negative
    FN = np.sum(np.logical_and(np.equal(y_true, cat2id[category]), np.not_equal(y_pred, cat2id[category])))
    # precision
    p = TP / (TP + FP)
    # recall
    r = TP / (TP + FN)
    # f1 score
    f1 = 2 * p * r / (p + r)
    return round(p, 3), round(r, 3), round(f1, 3)


# get scores for each corpus and print the macro f1 scores
def print_accuracy(y_true, y_pred, cat2id):
    p_Quran, r_Quran, f_Quran = accuracy_check(y_true, y_pred, "Quran", cat2id)
    p_NT, r_NT, f_NT = accuracy_check(y_true, y_pred,"NT", cat2id)
    p_OT, r_OT, f_OT = accuracy_check(y_true, y_pred,"OT", cat2id)
    # macro precision
    p_macro = round((p_Quran + p_NT + p_OT) / 3, 3)
    # macro recall
    r_macro = round((r_Quran + r_NT + r_OT) / 3, 3)
    # macro f1
    f_macro = round((f_Quran + f_NT + f_OT) / 3, 3)
    # print macro f1 scores
    print(f_macro)
    # store each score
    accuracy_list = [str(p_Quran), str(r_Quran), str(f_Quran), str(p_NT), str(r_NT), str(f_NT), str(p_OT), str(r_OT),
                     str(f_OT), str(p_macro), str(r_macro), str(f_macro)]
    return accuracy_list


# check incorrect predictions
def check_incorrect(predictions, true_values):
    incorrect = []
    correct = []
    for i in range(len(predictions)):
        if predictions[i] != true_values[i]:
            incorrect.append(i)
    return incorrect


# improve baseline model by removing stopwords
def improved_preprocess_data1(data, stopwords_list):
    chars_to_remove = re.compile(f'[{string.punctuation}]')
    documents = []
    categories = []
    vocab = set([])
    lines = data.split("\n")
    for line in lines:
        line = line.strip()
        if line:
            category, text = line.split("\t")
            # remove punctuation and casefold
            words_list = chars_to_remove.sub("", text).lower().split()
            # remove stopwords
            words = list(filter(lambda x: x not in stopwords_list, words_list))
            for word in words:
                vocab.add(word)
            documents.append(words)
            categories.append(category)
    return documents, categories, vocab


# improve baseline model by using top 5000 features with the highest MI scores
def improved_preprocess_data2(data, MI_Quran, MI_NT, MI_OT):
    X, Y, Z = Counter(MI_Quran), Counter(MI_NT), Counter(MI_OT)
    total_MI = dict(X + Y + Z)
    MI_list = []
    # select top 5000 features with the highest MI scores
    for i in sorted(total_MI.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5000]:
        MI_list.append(i[0])
    chars_to_remove = re.compile(f'[{string.punctuation}]')
    documents = []
    categories = []
    vocab = set([])
    lines = data.split("\n")
    for line in lines:
        line = line.strip()
        if line:
            category, text = line.split("\t")
            # remove punctuation and casefold
            words_list = chars_to_remove.sub("", text).lower().split()
            # select words in the top 5000 features
            words = [word for word in words_list if word in MI_list]
            for word in words:
                vocab.add(word)
            documents.append(words)
            categories.append(category)
    return documents, categories, vocab


# improve baseline model by using top 5000 features with the highest Chi_square scores
def improved_preprocess_data3(data, Chi_square_Quran, Chi_square_NT, Chi_square_OT):
    X, Y, Z = Counter(Chi_square_Quran), Counter(Chi_square_NT), Counter(Chi_square_OT)
    total_Chi_square = dict(X + Y + Z)
    Chi_square_list = []
    # select top 5000 features with the highest Chi_square scores
    for i in sorted(total_Chi_square.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:5000]:
        Chi_square_list.append(i[0])
    chars_to_remove = re.compile(f'[{string.punctuation}]')
    documents = []
    categories = []
    vocab = set([])
    lines = data.split("\n")
    for line in lines:
        line = line.strip()
        if line:
            category, text = line.split("\t")
            # remove punctuation and casefold
            words_list = chars_to_remove.sub("", text).lower().split()
            # select words in the top 5000 features
            words = [word for word in words_list if word in Chi_square_list]
            for word in words:
                vocab.add(word)
            documents.append(words)
            categories.append(category)
    return documents, categories, vocab


if __name__ == "__main__":
    # read the results file and qrel file
    system_results = pd.read_csv("system_results.csv", header=0, sep=",")
    qrels = pd.read_csv("qrels.csv", header=0, sep=",")
    # operate evaluation
    ir_eval = Eval(system_results, qrels)
    # write to ir_eval.csv file
    f1 = "ir_eval.csv"
    with open(f1, "a+") as file:
        file.write("system_number,query_number,P@10,R@50,r-precision,AP,nDCG@10,nDCG@20" + "\n")
    for index, row in ir_eval.iterrows():
        with open(f1, "a+") as file:
            file.write(str(int(row["system_number"])) + "," + str(index) + "," + "{:.3f}".format(row["P@10"]) + \
                       "," "{:.3f}".format(row["R@50"]) + "," + "{:.3f}".format(row["r-precision"]) + \
                       "," "{:.3f}".format(row["AP"]) + "," + "{:.3f}".format(row["nDCG@10"]) + "," + \
                       "{:.3f}".format(row["nDCG@20"]) + "\n")
    #read stopwords collection
    englishTS = open("englishST.txt")
    stopwords = englishTS.readlines()
    stopwords_list = []
    for word in stopwords:
        stopwords_list.append(word.replace("\n", ""))
    # read train_and_dev.tsv file
    data = open("train_and_dev.tsv").read()
    # divide into three corpora
    corpus_Quran, corpus_NT, corpus_OT, Quran_dic, NT_dic, OT_dic = Preprocess_and_divide(data,stopwords_list)
    # compute MI and Chi_square score
    X, Y, Z = Counter(Quran_dic), Counter(NT_dic), Counter(OT_dic)
    total_dic = dict(X + Y + Z)
    MI_Quran, Chi_square_Quran = Corpus_scores(total_dic, Quran_dic, corpus_Quran, corpus_NT, corpus_OT)
    MI_OT, Chi_square_OT = Corpus_scores(total_dic, OT_dic, corpus_Quran, corpus_NT, corpus_OT)
    MI_NT, Chi_square_NT = Corpus_scores(total_dic, NT_dic, corpus_Quran, corpus_NT, corpus_OT)
    # print top 10 words with highest score for each corpus
    print(sorted(MI_Quran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(Chi_square_Quran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(MI_OT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(Chi_square_OT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(MI_NT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])
    print(sorted(Chi_square_NT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:10])

    # run LDA model on three corpora
    lda, topic_dic_Quran, topic_dic_NT, topic_dic_OT = LDA_model(corpus_Quran, corpus_NT, corpus_OT)
    # rank the topics for each corpus
    topic_ranked_NT = sorted(topic_dic_NT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:1]
    topic_ranked_OT = sorted(topic_dic_OT.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:1]
    topic_ranked_Quran = sorted(topic_dic_Quran.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:1]
    for topic in topic_ranked_NT:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))
    for topic in topic_ranked_OT:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))
    for topic in topic_ranked_Quran:
        print("topic_id: " + str(topic[0]) + ", score: " + str(topic[1]))
        print(lda.print_topic(topic[0]))

    # read train data
    training_data = open("train_and_dev.tsv").read()
    # preprocessing
    preprocessed_data, categories, vocab = preprocess_data(training_data)
    # divide train and development set
    preprocessed_training_data, training_categories, preprocessed_dev_data, dev_categories, dev_index = Train_Dev_split(preprocessed_data, categories)
    # convert the vocab to a word id lookup dictionary
    word2id = {}
    for word_id, word in enumerate(vocab):
        word2id[word] = word_id
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    # convert train set to BOW
    X_train = convert_to_bow_matrix(preprocessed_training_data, word2id)
    y_train = [cat2id[cat] for cat in training_categories]
    # train SVC model on train set
    model = sklearn.svm.SVC(C=1000, random_state=0)
    model.fit(X_train, y_train)
    # convert development set to BOW
    X_dev = convert_to_bow_matrix(preprocessed_dev_data, word2id)
    y_dev = [cat2id[cat] for cat in dev_categories]
    # make predictions for train and development set
    y_train_predictions = model.predict(X_train)
    y_dev_predictions = model.predict(X_dev)
    # check incorrect predictions for development set
    incorrect = check_incorrect(y_dev_predictions, y_dev)
    for i in incorrect[:3]:
        print(preprocessed_data[dev_index[i]])
        print("predicted category: " + list(cat2id.keys())[list(cat2id.values()).index(y_dev_predictions[i])])
        print("true category: " + list(cat2id.keys())[list(cat2id.values()).index(y_dev[i])])

    # read test data
    testing_data = open("test.tsv").read()
    # preprocessing
    preprocessed_test_data, test_categories, test_vocab = preprocess_data(testing_data)
    # convert test set to BOW
    X_test = convert_to_bow_matrix(preprocessed_test_data, word2id)
    y_test = [cat2id[cat] for cat in test_categories]
    # make prediction for test set
    y_test_predictions = model.predict(X_test)
    # print baseline macro-f1 scores
    print("baseline:")
    base_train = print_accuracy(y_train, y_train_predictions, cat2id)
    base_dev = print_accuracy(y_dev, y_dev_predictions, cat2id)
    base_test = print_accuracy(y_test, y_test_predictions, cat2id)

    # improve the model by choosing C=10
    model = sklearn.svm.SVC(C=10, random_state=0)
    model.fit(X_train, y_train)
    y_train_predictions = model.predict(X_train)
    y_dev_predictions = model.predict(X_dev)
    y_test_predictions = model.predict(X_test)
    # print macro-f1 scores for C=10
    print("c=10:")
    imp_train = print_accuracy(y_train, y_train_predictions, cat2id)
    imp_dev = print_accuracy(y_dev, y_dev_predictions, cat2id)
    imp_test = print_accuracy(y_test, y_test_predictions, cat2id)

    # improve the model by choosing C=100
    model = sklearn.svm.SVC(C=100, random_state=0)
    model.fit(X_train, y_train)
    y_train_predictions = model.predict(X_train)
    y_dev_predictions = model.predict(X_dev)
    y_test_predictions = model.predict(X_test)
    # print macro-f1 scores for C=100
    print("c=100:")
    print_accuracy(y_train, y_train_predictions, cat2id)
    print_accuracy(y_dev, y_dev_predictions, cat2id)
    print_accuracy(y_test, y_test_predictions, cat2id)

    # improve the model by choosing C=5000
    model = sklearn.svm.SVC(C=5000, random_state=0)
    model.fit(X_train, y_train)
    # make predictions
    y_train_predictions = model.predict(X_train)
    y_dev_predictions = model.predict(X_dev)
    y_test_predictions = model.predict(X_test)
    # print macro-f1 scores for C=5000
    print("c=5000:")
    print_accuracy(y_train, y_train_predictions, cat2id)
    print_accuracy(y_dev, y_dev_predictions, cat2id)
    print_accuracy(y_test, y_test_predictions, cat2id)

    # using LogisticRegression on OneVsRestClassifier
    # train the model
    model = OneVsRestClassifier(LogisticRegression(random_state=0))
    clf = model.fit(X_train, y_train)
    clf.fit(X_train, y_train)
    # make predictions
    y_train_predictions = clf.predict(X_train)
    y_dev_predictions = clf.predict(X_dev)
    y_test_predictions = clf.predict(X_test)
    # print macro-f1 scores for logistic regression
    print("logistic:")
    print_accuracy(y_train, y_train_predictions, cat2id)
    print_accuracy(y_dev, y_dev_predictions, cat2id)
    print_accuracy(y_test, y_test_predictions, cat2id)

    # using DecisionTreeClassifier
    # train the model
    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    # make predictions
    y_train_predictions = clf.predict(X_train)
    y_dev_predictions = clf.predict(X_dev)
    y_test_predictions = clf.predict(X_test)
    # print macro-f1 scores for decision tree
    print("random forest:")
    print_accuracy(y_train, y_train_predictions, cat2id)
    print_accuracy(y_dev, y_dev_predictions, cat2id)
    print_accuracy(y_test, y_test_predictions, cat2id)

    # improve the model by removing stopwords
    # read train data
    training_data = open("train_and_dev.tsv").read()
    # preprocessing
    preprocessed_data, categories, vocab = improved_preprocess_data1(training_data, stopwords_list)
    # divide train and development set
    preprocessed_training_data, training_categories, preprocessed_dev_data, dev_categories, dev_index = Train_Dev_split(
        preprocessed_data, categories)
    # convert the vocab to a word id lookup dictionary
    word2id = {}
    for word_id, word in enumerate(vocab):
        word2id[word] = word_id
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    # convert train set to BOW
    X_train = convert_to_bow_matrix(preprocessed_training_data, word2id)
    y_train = [cat2id[cat] for cat in training_categories]
    # train SVC model on train set
    model = sklearn.svm.SVC(C=1000, random_state=0)
    model.fit(X_train, y_train)
    # convert development set to BOW
    X_dev = convert_to_bow_matrix(preprocessed_dev_data, word2id)
    y_dev = [cat2id[cat] for cat in dev_categories]
    # make predictions for train and development set
    y_train_predictions = model.predict(X_train)
    y_dev_predictions = model.predict(X_dev)
    # read test data
    testing_data = open("test.tsv").read()
    # preprocessing
    preprocessed_test_data, test_categories, test_vocab = improved_preprocess_data1(testing_data, stopwords_list)
    # convert test set to BOW
    X_test = convert_to_bow_matrix(preprocessed_test_data, word2id)
    y_test = [cat2id[cat] for cat in test_categories]
    # make predictions for test set
    y_test_predictions = model.predict(X_test)
    # print macro-f1 scores for removing stopwords
    print("remove stopwords:")
    print_accuracy(y_train, y_train_predictions, cat2id)
    print_accuracy(y_dev, y_dev_predictions, cat2id)
    print_accuracy(y_test, y_test_predictions, cat2id)

    # improve the model by using top 5000 features with the highest MI scores
    # read train data
    training_data = open("train_and_dev.tsv").read()
    # preprocessing
    preprocessed_data, categories, vocab = improved_preprocess_data2(training_data, MI_Quran, MI_NT, MI_OT)
    # divide train and development set
    preprocessed_training_data, training_categories, preprocessed_dev_data, dev_categories, dev_index = Train_Dev_split(
        preprocessed_data, categories)
    # convert the vocab to a word id lookup dictionary
    word2id = {}
    for word_id, word in enumerate(vocab):
        word2id[word] = word_id
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    # convert train set to BOW
    X_train = convert_to_bow_matrix(preprocessed_training_data, word2id)
    y_train = [cat2id[cat] for cat in training_categories]
    # train SVC model on train set
    model = sklearn.svm.SVC(C=1000, random_state=0)
    model.fit(X_train, y_train)
    # convert development set to BOW
    X_dev = convert_to_bow_matrix(preprocessed_dev_data, word2id)
    y_dev = [cat2id[cat] for cat in dev_categories]
    # make predictions for train and development set
    y_train_predictions = model.predict(X_train)
    y_dev_predictions = model.predict(X_dev)
    # read test data
    testing_data = open("test.tsv").read()
    # preprocessing
    preprocessed_test_data, test_categories, test_vocab = improved_preprocess_data2(testing_data, MI_Quran, MI_NT, MI_OT)
    # convert test set to BOW
    X_test = convert_to_bow_matrix(preprocessed_test_data, word2id)
    y_test = [cat2id[cat] for cat in test_categories]
    # make predictions for test set
    y_test_predictions = model.predict(X_test)
    # print macro-f1 scores for using top 5000 features with the highest MI scores
    print("top 5000 features with the highest MI scores:")
    print_accuracy(y_train, y_train_predictions, cat2id)
    print_accuracy(y_dev, y_dev_predictions, cat2id)
    print_accuracy(y_test, y_test_predictions, cat2id)

    # improve the model by using top 5000 features with the highest Chi_square scores
    # read train data
    training_data = open("train_and_dev.tsv").read()
    # preprocessing
    preprocessed_data, categories, vocab = improved_preprocess_data3(training_data, Chi_square_Quran, Chi_square_NT, Chi_square_OT)
    # divide train and development set
    preprocessed_training_data, training_categories, preprocessed_dev_data, dev_categories, dev_index = Train_Dev_split(
        preprocessed_data, categories)
    # convert the vocab to a word id lookup dictionary
    word2id = {}
    for word_id, word in enumerate(vocab):
        word2id[word] = word_id
    cat2id = {}
    for cat_id, cat in enumerate(set(categories)):
        cat2id[cat] = cat_id
    # convert train set to BOW
    X_train = convert_to_bow_matrix(preprocessed_training_data, word2id)
    y_train = [cat2id[cat] for cat in training_categories]
    # train SVC model on train set
    model = sklearn.svm.SVC(C=1000, random_state=0)
    model.fit(X_train, y_train)
    # convert development set to BOW
    X_dev = convert_to_bow_matrix(preprocessed_dev_data, word2id)
    y_dev = [cat2id[cat] for cat in dev_categories]
    # make predictions for train and development set
    y_train_predictions = model.predict(X_train)
    y_dev_predictions = model.predict(X_dev)
    # read test data
    testing_data = open("test.tsv").read()
    # preprocessing
    preprocessed_test_data, test_categories, test_vocab = improved_preprocess_data3(testing_data, Chi_square_Quran, Chi_square_NT, Chi_square_OT)
    # convert test set to BOW
    X_test = convert_to_bow_matrix(preprocessed_test_data, word2id)
    y_test = [cat2id[cat] for cat in test_categories]
    # make predictions for test set
    y_test_predictions = model.predict(X_test)
    # print macro-f1 scores for using top 5000 features with the highest Chi_square scores
    print("top 5000 features with the highest Chi_square scores:")
    print_accuracy(y_train, y_train_predictions, cat2id)
    print_accuracy(y_dev, y_dev_predictions, cat2id)
    print_accuracy(y_test, y_test_predictions, cat2id)

    # write to classification.csv
    with open("classification.csv", "a+") as file:
        file.write("system,split,p-quran,r-quran,f-quran,p-ot,r-ot,f-ot,p-nt,r-nt,f-nt,p-macro,r-macro,f-macro" + "\n")
        file.write("baseline,train," + ",".join(base_train) + "\n")
        file.write("baseline,dev," + ",".join(base_dev) + "\n")
        file.write("baseline,test," + ",".join(base_test) + "\n")
        file.write("improved,train," + ",".join(imp_train) + "\n")
        file.write("improved,dev," + ",".join(imp_dev) + "\n")
        file.write("improved,test," + ",".join(imp_test) + "\n")

