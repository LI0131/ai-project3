# ai-project3
Recurrent Neural Network for Natural Language Processing

# Dev Setup
`python3 app.py`

# About This Project

## Tokenization

Tokenization is the process of transforming an input file into a format which is acceptable for creating embeddings. We rely heavily on the NLTK package to create our tokenized input. We use the `word_tokenize` to take convert each csv row into an array of tokens. These tokens can be individual words, punctation, or word pieces. We compute the frequency of each token in the csv file to create our vocabulary. Our vocabulary is composed on the 8000 most frequent tokens in the csv file. We restrict the vocabulary so that we limit the effects of the vanishing gradient problem. With too many tokens in our vocab, the softmax function divides the probability associated with the perdiction over too many vocabulary tokens. We remove stopwords, because they add no or minimal meaning to sentence, and as such can be neglected in creating our perdiction. We then add START and END tokens to each phrase to tell the model when each phrase begins and end. WE add UNK tokens to fill the place of words that are not in our vocabulary. We then use the Gensim package's Word2Vec class to create embeddings which correlate to each string.