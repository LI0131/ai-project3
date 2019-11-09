import logging
import numpy as np 
from scipy.special import softmax

logging.basicConfig(level=logging.INFO)


class ExampleGenerator():

    def __init__(self, w2v, tokens):
        self.w2v = w2v
        self.tokens = tokens
        self.pointer = 0

    def generate_example(self):
        example = [self.w2v.wv[word] for word in self.tokens[self.pointer]]
        target = example[2:]
        example = example[:len(example) - 2]
        self.pointer += 1
        return (example, target)

    def __call__(self):
        return self.generate_example()


class RNNModel():

    def __init__(self, word_dim, hidden_dim, vocab):
        # init weight matrices
        self.U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),(word_dim, hidden_dim))
        self.W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim),(hidden_dim, hidden_dim))
        # init helper values
        self.matStore = []
        self.word_dim = word_dim
        self.vocab = vocab

    def propagate_forward(self, sentence):
        self.matStore = np.zeros((len(sentence), self.word_dim))
        for i in range(1, len(sentence)):
            hidden = np.tanh(np.dot(self.U, sentence[i]) + np.dot(self.W, self.matStore[i-1]))
            self.matStore[i] = hidden
            output = softmax(np.dot(hidden, self.V))
            logging.info(f'Output for index {i}: {self.predict(output)}')

    def predict(self, output):
        return self.vocab[np.argmax(output)]

    def __call__(self, embedded_input):
        tokens, self.target = embedded_input
        self.propagate_forward(tokens)
