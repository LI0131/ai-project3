import csv
import nltk
import logging
import operator

from nltk.tokenize import RegexpTokenizer, word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec

nltk.download(['punkt', 'stopwords'])
logging.basicConfig(level=logging.INFO)

tokenizer = RegexpTokenizer(r'\w+')
unknown_token = 'UNK'


def run():
    data_file = open('data/rnnDataset.csv')
    data = csv.reader(data_file, delimiter='\n')
    tokens = []

    logging.info(f'Generating alpha-numeric tokens from {data_file.name}...')
    for line in data:
        string = str(line[0])
        tokens.append(
            # [x.lower() for x in filter(lambda x: x.isalnum(), word_tokenize(string))]
            # [x.lower() for x in filter(lambda x: tokenizer.tokenize(x), word_tokenize(string))]
            # tokenizer.tokenize(string)
        )

    print(tokens)

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

    print(tokens)

    model = Word2Vec(tokens, min_count=1, size=100, window=5)
    print(model)


if __name__ == '__main__':
    run()
