import networkx as nx
import nltk
import collections
import itertools
from nltk.tokenize import word_tokenize
import re
from nltk.stem import WordNetLemmatizer
import numpy as np
from glob import glob

lemmatizer = WordNetLemmatizer()


class readFile:

    @staticmethod
    def read_file_folder(path, regular_expression=None, encoding="utf-8"):
        """
        It will match files on a path based on a regular expression
        :param path:
        :param regular_expression: Eg r".+/((\w)\d{2,4}).txt
                 If given, will extract the name from the file name.
                 Otherwise, it'll take the file name as the name
                 encoding  : use 'windows-1252' for html type reading
        :return: dictionary where keys are the document name and values the
                text in the documents
        """
        files = glob(path)
        data = {}
        for file in files:
            if regular_expression is not None:
                file_name = re.match(regular_expression, file).group(1)
            else:
                file_name = file
                # category = re.match(r".+/((\w)\d{2,4}).txt", file).group(2)
            f = open(file, 'r', encoding=encoding)
            text = f.read()

            data[file_name] = text
            f.close()
        return data

class preprocessing:
    @staticmethod
    def apply_re(x):
        """
        Apply regular expressions to clean up the text
        :param x: string input
        :return: string after substitution applied
        """
        x = re.sub("\\n|\\t", " ", x)  # remove new line and tab symbols
        x = re.sub("((?<=[A-Za-z])'s)", " ", x)  # replace 's preceded by a word by a space. Eg: Ana's -> Ana_
        x = re.sub("\(|\)|!|-|\[|\]|;|:|<|>|\.|,|/|\?|@|#|\$|%|\^|\&|\*|_|\~", " ",
                   x)  # replace punctuaction signs by space. \ needs to be implemented
        return x

    @staticmethod
    def remove_single_characters(x):
        return ' '.join([w for w in word_tokenize(x) if len(w) > 1])

    @staticmethod
    def get_only_words(x):
        """
        :param x:
        :return:
        """
        clean_x = re.sub("[^a-zA-Z]", " ", x)
        return clean_x

    @staticmethod
    def to_lower_case(x):
        """

        :param x:
        :return:
        """
        return x.lower()

    @staticmethod
    def filter_tags(x, persist_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
        """

        :param x:
        :param persist_postags:
        :return:
        """
        eq = {'NOUN': "NN",
              'ADJ': "JJ",
              'VERB': "VB",
              'ADV': "RB"}
        persistPostags_eq = [eq[x] for x in persist_postags]

        def to_tag(tag):
            return any(tag.startswith(tags_eq) for tags_eq in persistPostags_eq)

        return ' '.join([word for (word, tag) in nltk.pos_tag(word_tokenize(x)) if to_tag(tag)])

    @staticmethod
    def remove_single_characters(x):
        """

        :param x:
        :return:
        """
        return ' '.join([w for w in word_tokenize(x) if len(w) > 1])

    @staticmethod
    def lemmatization(x):
        """

        :param x:
        :return:
        """

        result = ' '.join([lemmatizer.lemmatize(word) for word in word_tokenize(x)])
        return result

    def clean_text(self, x):
        """

        :param x:
        :return:
        """
        result = self.apply_re(x)
        result = self.remove_single_characters(result)
        result = self.get_only_words(result)
        result = self.to_lower_case(result)
        result = self.filter_tags(result, persist_postags=['NOUN', 'ADJ'])
        result = self.lemmatization(result)
        return word_tokenize(result)


class createGraph:
    """
    This class creates a weigthed indirected graph based on word co-ocurrence.
    Attributes
    ----------
    global_path : This is the path were the documents are.
            The path can be unix-like regular expressions.
    window_size : size of the window for selecting co-ocurrence words
    """

    def __init__(self, global_path, window_size, regular_expression=None):
        self.global_path = global_path
        self.regular_expression = regular_expression
        self.window_size = window_size
        self.remove_nodes_itself = True

    def get_window_list(self, doc_dict):
        # we can assert that the first element in the list should be present in the previous list
        """
        Parameters
        ----------
        doc : dictionary of documents where key is the document name and value
        \t\t is a list of words in that document.
        """
        window_size = self.window_size
        data = []
        for doc_name, doc in doc_dict.items():
            for i, j in enumerate(doc):
                if i + window_size <= len(doc):
                    data.append(doc[i:i + window_size])

        return np.array(data)

    def check_words_presence(self, ll, words):
        """
        Given an input list. Checks wheter all the words are in the list or not.
        """
        return all(w in ll for w in words)

    def get_words_coocurrence(self, window_list, word_list):
        """
        """
        co_ocurrence_words = [x for x in window_list if self.check_words_presence(x, word_list)]
        return len(co_ocurrence_words)

    def read_file_preprocess(self):
        """
        Returns
        -------
        Dictionary with the keys the name of the file (clened by
        a regular expression, if given) and values the processed text.

        """
        dt = readFile.read_file_folder(self.global_path)

        dt_cleaned = {}
        preprocessing_instance = preprocessing()
        for k, v in dt.items():
            dt_cleaned[k] = preprocessing_instance.clean_text(v)
        return dt_cleaned

    @staticmethod
    def flat_list(ll):
        """
        This method converts a list of list into a single list
        """
        return [w for x in ll for w in x]

    def co_ocurrence_list(self, document):
        """
        Given a list of words and window size, generate the co-occurence list
        and the times two words co-appear.

        Returns
        -------
        List of tuples (w1,w2). Each element of the list represent an co-ocurrence of the two words.
        The tuples are in alphabetical order, i.e w1<w2.

        """
        window_list = self.get_window_list(document)
        results = []
        for words_list in window_list:
            words_list.sort()
            results += list(set(itertools.combinations(words_list, 2)))
        if self.remove_nodes_itself:  # remove edges that goes to one edge to itself
            results = [x for x in results if x[0] != x[1]]
        return results

    def set_edges_nodes(self, document):
        """

        """
        co_ocurrence_list = self.co_ocurrence_list(document)
        self.edges = collections.Counter(co_ocurrence_list)
        self.nodes = list(set(itertools.chain.from_iterable(co_ocurrence_list)))
        return True

    def make_graph(self):
        """
        Run the pipeline:
        \t 1. Read the data from the global path
        \t 2. and calculate the edges and nodes
        """
        doc_dict = self.read_file_preprocess()
        self.set_edges_nodes(doc_dict)

    def create_graph(self):
        """
        Create the graph as a python netowrkx structure
        :return: graph of word co-ocurrence
        """
        self.make_graph()

        G = nx.MultiDiGraph()
        G.add_nodes_from(self.nodes)
        for k, v in self.edges.items():
            G.add_edge(k[0], k[1], weight=v)
        nx.set_node_attributes(G, dict(zip(list(G.nodes), [1] * len(G.nodes))), 'WS')
        return G

