import os
from datetime import datetime
import numpy as np
import pandas as pd
from tqdm import trange
import sys
import random
from gensim.corpora import Dictionary


class LDA():
    """
    Latent Dirichlet Allocation (LDA) Model.
    Developed for didactic purposes.

    Parameters
    ----------
    num_topics : int
        The number of topics we want to extract from our LDA model. Lower numbers will improve performance.
    corpus : list
        List comprehension of the corpus.
        Each sublist in the list represents a document, and contains a series of strings (the words of the document).
    alpha : int
        Parameter of the exchangeable Dirichlet prior on the proportions of topics per document.
        Higher values of alpha will produce more homogeneous topic mixtures in each document, lower values will produce more "concentrated" mixtures around a given topic.
    beta : int
        Parameter of the exchangeable Dirichlet prior on the proportions of topics per word.
    epochs : int
        Number of training epochs. Higher values will increase accuracy of the model.
    no_below : int
        (optional) Exclude words that appear less than `no_below` times from the training data.
    no_below : float
        (optional) Exclude words that appear more than `no_above`% of the documents from the training data.
    """
    def __init__(self, num_topics, corpus, alpha, beta, epochs, no_below=0, no_above=1.0):
        """
        Initialize the LDA object and train it.
        """
        self.num_topics = num_topics
        topics = list(range(self.num_topics))
        self.corpus = corpus
        self.alpha = alpha
        self.beta = beta

        # filter the words
        self.dictionary = Dictionary(self.corpus)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        filtered_words = list(self.dictionary.token2id.keys())
        # transform words in integers for faster computations
        self.numerical_dict = dict(zip(filtered_words, range(len(filtered_words))))
        self.corpus = [[self.numerical_dict[e] for e in speech if e in filtered_words] for speech in self.corpus]

        # Vocabulary V
        self.V = sorted(set([e for speech in self.corpus for e in speech]))
        # Ns - length of each document
        self.N = [len(speech) for speech in self.corpus]
        # M - number of documents in corpus
        self.M = len(self.N)

        # Randomly assign each word in each document to a random topic
        self.Z = [[random.sample(topics, 1)[0] for word in speech] for speech in self.corpus]

        # Create the N_{d,topic} matrix. It dispays the occurrence of each topic (col) in each document (row).
        self.N_d_topic = []
        for document in self.Z:
            N_d = []
            for topic in topics:
                N_d.append(sum([1 for e in document if e == topic]))
            self.N_d_topic.append(N_d)
        self.N_d_topic = np.array(self.N_d_topic)

        # Create the N_{word,topic} matrix. It displays the occurrence of each topic(col) in each word (row).
        flat_instances = [e for sublist in self.corpus for e in sublist]
        flat_Z = [e for sublist in self.Z for e in sublist]
        self.N_word_topic = {}
        for v in self.V:
            self.N_word_topic[v] = [0 for e in topics]
        for word, assigned_topic in zip(flat_instances, flat_Z):
            for topic in topics:
                if assigned_topic == topic:
                    self.N_word_topic[word][topic] += 1
        self.N_word_topic = pd.DataFrame(self.N_word_topic).T.values

        # Create the N_{topic} vector. It displays the frequency of each topic. Initially, should be uniformly distributed.
        self.N_topic = [0 for topic in topics]
        for assigned_topic in flat_Z:
            for topic in topics:
                if assigned_topic == topic:
                    self.N_topic[topic] += 1
        self.N_topic = np.array(self.N_topic)

        # TRAINING VIA GIBBS SAMPLING
        tic = datetime.utcnow() #store starting time to keep track of training time
        self.topic_frequencies = []
        self.thetas = []
        self.phis = []
        for iteration in trange(epochs, file=sys.stdout, desc='LDA'):
            for i_doc in range(self.M):
                for i_word in range(len(self.corpus[i_doc])):
                    word = self.corpus[i_doc][i_word]
                    topic_assigned = self.Z[i_doc][i_word]

                    self.N_d_topic[i_doc, topic_assigned] -= 1
                    self.N_word_topic[word, topic_assigned] -= 1
                    self.N_topic[topic_assigned] -= 1

                    percs = []
                    for topic in topics:
                        perc = ((self.N_d_topic[i_doc, topic] + alpha) / np.sum(self.N_d_topic[i_doc] + alpha)
                                * (self.N_word_topic[word, topic] + beta) / np.sum(self.N_word_topic.T[topic] + beta))
                        if perc < 0:
                            raise Exception("perc < 0")
                        percs.append(perc)

                    s = np.sum(percs)
                    percs = [e/s for e in percs]
                    new_topic = np.random.choice(topics, 1, p=percs)[0]

                    self.N_d_topic[i_doc, new_topic] += 1
                    self.N_word_topic[word, new_topic] += 1
                    self.N_topic[new_topic] += 1

                    self.Z[i_doc][i_word] = new_topic

            # store N_topic every 10 iterations
            if iteration % 10 == 0:
                self.topic_frequencies.append(self.N_topic.copy())

                somma_theta = self.N_d_topic.sum(axis=1).reshape((-1,1))
                theta = np.divide(self.N_d_topic, somma_theta)
                self.thetas.append(theta.copy())

                somma_phi = self.N_word_topic.sum(axis=0).reshape((-1,1))
                phi = np.divide(self.N_word_topic.T, somma_phi)
                self.phis.append(phi.copy())
        
        toc = datetime.utcnow()
        print(f"Training completed in {toc-tic}")

    def show_topics(self, top_n_words=5):
        """
        Show the top `top_n_words` associated with each topic.
        """
        inv_dict = {v: k for k, v in self.numerical_dict.items()}
        for topic in range(self.num_topics):
            print(f"Main words for topic {topic}:")
            word_indices = pd.DataFrame(self.N_word_topic).sort_values(by=topic, ascending = False).index[:top_n_words].to_list()
            print([inv_dict[e] for e in word_indices])
            print()
