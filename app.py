import nltk
import logging
import operator
import pandas as pd

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

from model import ExampleGenerator, RNNModel

nltk.download(['punkt', 'stopwords'])
logging.basicConfig(level=logging.INFO)

data_file = 'data/rnnDataset.csv'
unknown_token = 'UNK'
vector_size = 10


def run():
    data = pd.read_csv(data_file, engine='python')
    tokens = []

    logging.info(f'Generating alphabetic tokens from {data_file}...')
    for (num, line) in data.iterrows():
        string = str(line[0])
        tokens.append(
            [x.lower() for x in word_tokenize(string)]
        )

    logging.info('Calculating Frequency of tokens...')
    word_to_freq = {}
    for sentence in tokens:
        for word in sentence:
            if word in word_to_freq:
                word_to_freq[word] += 1
            else:
                word_to_freq[word] = 1

    logging.info('Truncating vocabulary to 8000 most frequent tokens...')
    word_to_freq = sorted(word_to_freq.items(), key=operator.itemgetter(1), reverse=True)

    logging.info('Removing stopwords...')
    word_to_freq = [[k,v] for [k,v] in word_to_freq if k not in stopwords.words('english')]

    vocab = []
    for i in range(8000):
        vocab.append(word_to_freq[i][0])

    logging.info('Append START and END tokens dropping non-vocab words...')
    for i in range(len(tokens)):
        temp_sentence = ['START']
        temp_sentence.extend([word if word in vocab else unknown_token for word in tokens[i]])
        temp_sentence.append('END')
        tokens[i] = temp_sentence

    logging.info(f'Here are 10 example tokenized input strings: {[tokens[i] for i in range(10)]}')

    model = Word2Vec(tokens, min_count=1, size=vector_size, window=5)

    logging.info(f'Here are 10 example embedded strings: {[model.wv[word] for word in tokens[i] for i in range(10)]}')

    example_generator = ExampleGenerator(model, tokens)
    RNN = RNNModel(vector_size, vector_size, vocab)

    for i in range(5):
        example = example_generator()
        RNN(example)
        

if __name__ == '__main__':
    run()
