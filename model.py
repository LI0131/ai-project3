from scipy.special import softmax
import numpy as np 


class ExampleGenerator():

    def __init__(self, w2v, tokens):
        self.w2v = w2v
        self.tokens = tokens
        self.pointer = 0

    def generate_example(self):
        example = [self.w2v.wv[word] for word in self.tokens[self.pointer]]
        target_token = self.tokens[self.pointer][len(example) - 2]
        target = example[len(example) - 2]
        example = list(filter(lambda x: x != target_token, self.tokens[self.pointer]))
        self.pointer += 1
        return (example, target)

    def __call__(self):
        return self.generate_example()


class RNNModel():

    def __init__(self, T, vocab):
        # init layers
        self.input = np.array([0]*100)
        self.hidden = [0]*100
        self.output = np.array([0]*(len(vocab) - 3)) # remove START, END, and UNK
        # init weight matrices
        self.U = np.random.normal(
            loc=0,
            scale=((len(self.input)+len(self.hidden))**(-0.5)),
            size=(len(self.input), len(self.hidden))
        )
        self.W = np.random.normal(
            loc=0,
            scale=((len(self.hidden)+len(self.hidden))**(-0.5)),
            size=(len(self.hidden), len(self.hidden))
        )
        self.V = np.random.normal(
            loc=0,
            scale=((len(self.hidden)+len(self.output))**(-0.5)),
            size=(len(self.hidden), len(self.output))
        )
        # init helper values
        self.target = []
        self.matStore = []
        self.T = T
        self.vocab = vocab

    def propagate_forward(self):
        self.hidden = np.tanh(np.dot(self.input, self.U))
        for x in range(self.T):
            self.hidden = np.tanh(np.dot(self.hidden, self.W))
            self.matStore.append(self.hidden)
        self.output = softmax(np.dot(self.hidden, self.V))

    def predict(self):
        return self.vocab[np.argmax(self.output)]

    def __call__(self, embedded_input):
        tokens, self.target = embedded_input
        for token in tokens:
            self.input = token
            self.propagate_forward()
