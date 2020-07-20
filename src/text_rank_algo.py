from nltk.stem import WordNetLemmatizer
import numpy as np
from copy import copy
lemmatizer = WordNetLemmatizer()

class textRankAlgorithm:

    graph_ranking_attribute = "WS"

    def __init__(self, window_size, num_keywords, d_constant = 0.85):
        self.window_size = window_size
        self.num_keywords = num_keywords
        self.d_constant = d_constant


    def get_new_weight(self, G, word):
        """
        Given a graph and a word in that graph make one iteration
        and compute the new value for that word
        :param G:
        :param word:
        :return:
        """
        in_edges = list(G.in_edges(word))
        out_edges = list(G.out_edges(word))
        sum_values = [0]

        current_WS = G.nodes[word]['WS']
        if len(in_edges) > 0 and len(out_edges) > 0:
            for i_edge in in_edges:
                w_ij = G.get_edge_data(*i_edge, default=0)[0]['weight']
                w_jk_total = []
                for j_edge in out_edges:
                    w_jk = G.get_edge_data(*j_edge, default=0)[0]['weight']
                    w_jk_total.append(w_jk)
                w_jk_total = np.sum(w_jk_total)
                sum_values.append(current_WS * w_ij / w_jk_total)
        new_ws = (1 - self.d_constant) + self.d_constant * np.sum(sum_values)
        return [word, new_ws]


    def update_nodes_values(self,G, nodes_list):
        """
        Make one iteration of the method and give the new value for the
        nodes as a dictionary
        :param G:
        :param nodes_list:
        :return:
        """
        nodes_dict = dict(zip(nodes_list, [None] * len(nodes_list)))
        for node in nodes_list:
            word, new_w = self.get_new_weight(G, node)
            nodes_dict[word] = new_w
        return nodes_dict

    def update_edges_values(self,G):
        nodes_list = list(G.nodes)
        return self.update_nodes_values(G, nodes_list)

    @staticmethod
    def ge_ws(G):
        """
        Get the ranking values from the graph
        :param G:
        :return:
        """
        res = []
        for node in list(G.nodes):
            res.append(G.nodes[node]['WS'])
        return res

    @staticmethod
    def update_g( G, res):
        """

        :param G:
        :param res: dictionary with keys the words
            and values the ranking
        :return:
        """
        max_val = np.max(list(res.values()))
        for n, d in res.items():
            G.nodes[n].update({'WS': d / max_val})
        return G

    def update_graph(self, G, max_iters=100, threshold=0.0001):
        """
        If the graph keeps improving, update the graph
        :param G: input graph
        :param max_iters:
        :param threshold:
        :return:
        """
        copy_g = copy(G)
        for i in range(max_iters):
            if i > 0: last_dicc = copy(my_dicc)
            my_dicc = self.update_edges_values(G)
            if i > 0:
                if any(abs(v - my_dicc.get(k)) < threshold for k, v in last_dicc.items()):
                    # print("Breaking loop in iteration {}".format(i))
                    break
            copy_g = self.update_g(copy_g, dict(my_dicc))
        return copy_g

    @staticmethod
    def top_words(G, num_keywords):
        """
        Return the top word based on the graph ranking
        :param G:
        :param num_keywords:
        :return:
        """
        res = []
        for node in list(G.nodes):
            res.append([G.nodes[node]['WS'], node])
        res.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in res[:num_keywords]]

    def get_keywords(self, G):
        return self.top_words(self.update_graph(G), self.num_keywords)

