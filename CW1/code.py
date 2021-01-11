import re
from stemming.porter2 import stem
from xml.etree import cElementTree as ET
import math


#proximity search & phrase search
def phrase_index(term1, term2, dic, distacne):
    dic1 = dic[term1]
    dic2 = dic[term2]
    page_list = dic1.keys() & dic2.keys()
    query_list = []
    for page in page_list:
        flag = True
        list1 = [int(x) for x in dic1[page]]
        list2 = [int(x) for x in dic2[page]]
        for p1 in list1:
            #proximity search
            if distacne > 1:
                for index in list(range(-distacne, distacne + 1)):
                    if p1 + index in list2:
                        flag = False
                        query_list.append(int(page))
                        break
                if not flag:
                    break
            #phrase search
            else:
                if p1 + 1 in list2:
                    flag = False
                    query_list.append(int(page))
                    break
                if not flag:
                    break
    return query_list


#boolean search
def bool_search(list1, list2, flag1, flag2, flag3, total_list):
    query_list = []
    # and
    if flag1 == True and flag3 == False:
        query_list = [x for x in list1 if x in list2]
    # and not
    elif flag1 == True and flag3 == True:
        query_list = [x for x in list1 if x not in list2]
    # or
    elif flag2 == True and flag3 == False:
        query_list = list(set(list1).union(set(list2)))
    # or not
    elif flag2 == True and flag3 == True:
        query_list = list(set(list1).union(set(total_list).difference(set(list2))))
    return query_list


#index for single word
def word_index(term, dic):
    query_list = []
    for key in dic[term]:
        query_list.append(int(key))
    return query_list


#search on index
def query_request(queryinput):
    # preprocessing
    remove_chars = '[!"$%&\()*+,-./:;<=>?@[\\]^_{|}~]+|\r|\t'
    query_words = re.sub(remove_chars, " ", queryinput).lower()
    words = filter(lambda x: x not in stopwords2 and x != "", query_words.split(" "))
    query_terms = [stem(word) for word in words]
    # split query
    list1 = []
    list2 = []
    flag1 = False #flag for AND
    flag2 = False #flag for OR
    flag3 = False #flag for NOT
    for index in range(len(query_terms)):
        #A AND B
        if query_terms[index] == "and" and "not" not in query_terms:
            flag1 = True
            list1 = query_terms[:index]
            list2 = query_terms[index + 1:]
        #A OR B
        elif query_terms[index] == "or" and "not" not in query_terms:
            flag2 = True
            list1 = query_terms[:index]
            list2 = query_terms[index + 1:]
        elif query_terms[index] == "not" and index != 0:
            flag3 = True
            #A AND NOT B
            if query_terms[index - 1] == "and":
                flag1 = True
                list1 = query_terms[:index - 1]
                list2 = query_terms[index + 1:]
            #A OR NOT B
            elif query_terms[index - 1] == "or":
                flag2 = True
                list1 = query_terms[:index - 1]
                list2 = query_terms[index + 1:]
        elif query_terms[index] == "not" and index == 0:
            flag3 = True
            #NOT A AND B
            if "and" in query_terms:
                flag1 = True
                i = query_terms.index("and")
                list2 = query_terms[1:i]
                list1 = query_terms[i + 1:]
            #NOT A OR B
            elif "or" in query_terms:
                flag2 = True
                i = query_terms.index("or")
                list2 = query_terms[1:i]
                list1 = query_terms[i + 1:]
            #NOT A
            else:
                list1 = query_terms[1:]
    total_list = [x for x in range(1, len(term_dic))]
    if flag1 == False and flag2 == False and flag3 == False:
        # proximity search
        if len(query_terms) == 3:
            space = int(query_terms[0].strip("#"))
            query_list = phrase_index(query_terms[1], query_terms[2], term_dic, space)
        # phrase search
        elif len(query_terms) == 2:
            query_list = phrase_index(query_terms[0], query_terms[1], term_dic, 1)
        # index search
        elif len(query_terms) == 1:
            query_list = word_index(query_terms[0], term_dic)
    # boolean search
    else:
        #not
        if flag1 == False and flag2 == False and flag3 == True:
            if len(list1) == 1:
                query_list1 = word_index(list1[0], term_dic)
                query_list = list(set(query_list1).difference(set(total_list)))
            elif len(list1) == 2:
                query_list1 = phrase_index(list1[0], list1[1], term_dic, 1)
                query_list = list(set(query_list1).difference(set(total_list)))
        else:
            if len(list1) == 1:
                query_list1 = word_index(list1[0], term_dic)
            else:
                query_list1 = phrase_index(list1[0], list1[1], term_dic, 1)
            if len(list2) == 1:
                query_list2 = word_index(list2[0], term_dic)
            else:
                query_list2 = phrase_index(list2[0], list2[1], term_dic, 1)
            query_list = bool_search(query_list1, query_list2, flag1, flag2, flag3, total_list)
    return sorted(query_list)


if __name__ == "__main__":
    #preprocess stopwords
    englishTS = open("/Users/zhongshilei/Desktop/TTDS/CW1collection/englishST.txt")
    stopwords = englishTS.readlines()
    stopwords = [word.strip("\n") for word in stopwords]
    stopwords2 = [word for word in stopwords if word != "and" and word != "or" and word != "not"]
    #preprocess xml
    xmlstr = open('/Users/zhongshilei/Desktop/TTDS/CW1collection/trec.5000.xml',encoding='utf8').read()
    root = ET.fromstring(xmlstr)
    documents = {}
    for page in list(root):
        num = page.find('DOCNO').text
        headline = page.find('HEADLINE').text
        content = page.find('TEXT').text
        documents[num] = headline + " " + content
    #store positional inverted index in dictionary
    total_num = len(documents) #total document number
    term_dic = {} #dictionary of positional inverted index
    for num in documents:
        #preprocessing
        doc = documents[num]
        content_withoutoken = re.sub(r"[^\w]+", " ", doc).lower()
        words_list = content_withoutoken.split(" ")
        words = filter(lambda x: x not in stopwords and x != "", words_list)
        terms = [stem(word) for word in words]
        count = 0
        #store inverted index
        for term in terms:
            if term not in term_dic:
                term_dic[term] = {}
            if num not in term_dic[term]:
                term_dic[term][num] = []
                term_dic[term][num].append(str(count))
            else:
                term_dic[term][num].append(str(count))
            count += 1
    # write index.txt
    f1 = "index.txt"
    for term in term_dic:
        with open(f1, "a+") as file:
            file.write(term + ": " + str(len(term_dic[term])) + "\n")
            for doc in term_dic[term]:
                file.write("\t" + doc + ": " + ",".join(term_dic[term][doc]) + "\n")
    # write results.boolean.txt
    f2 = "results.boolean.txt"
    #preprocessing queries for boolean search
    queries_1 = open("/Users/zhongshilei/Desktop/TTDS/CW1collection/queries.boolean.txt")
    lines_1 = queries_1.readlines()
    query_terms_1 = []
    for index in range(len(lines_1)):
        query_words = " ".join(lines_1[index].split(" ")[1:])
        query_words = query_words.strip("\n")
        boolean_results = query_request(query_words)
        for doc in boolean_results:
            with open(f2, "a+") as file:
                file.write(str(index + 1) + "," + str(doc) + "\n")
    #preprocessing queries for rank
    queries = open("/Users/zhongshilei/Desktop/TTDS/CW1collection/queries.ranked.txt")
    lines = queries.readlines()
    query_terms = []
    for line in lines:
        query_words = re.sub(r"[^\w]+", " ", line).lower().strip("\n")
        words = filter(lambda x: x not in stopwords and x != "", query_words.split(" "))
        query_term = [stem(word) for word in words]
        query_terms.append(query_term)
    #store tfidf weighting in dictionary
    doc_dic = {}
    for query_num in range(len(query_terms)):
        doc_dic[query_num] = {}
        for word in query_terms[query_num][1:]:
            doc_num = query_num + 1
            list_docs = query_request(word)
            #document frequency
            df = len(term_dic[word])
            for doc in list_docs:
                if doc not in doc_dic[query_num]:
                    doc_dic[query_num][doc] = 0
                #term frequency
                tf = len(term_dic[word][str(doc)])
                w = (1 + math.log10(tf)) * math.log10(total_num / df)
                doc_dic[query_num][doc] += w
    # write results.ranked.txt
    f3 = "results.ranked.txt"
    for query_num in range(len(query_terms)):
        for ranked_doc in sorted(doc_dic[query_num].items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:150]:
            with open(f3, "a+") as file:
                file.write(str(query_num + 1) + "," + str(ranked_doc[0]) + "," + "{:.4f}".format(ranked_doc[1]) + "\n")


