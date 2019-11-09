# ai-project3
Recurrent Neural Network for Natural Language Processing

# Dev Setup
`python3 app.py`

# About This Project

## Tokenization

Tokenization is the process of transforming an input file into a format which is acceptable for creating embeddings. We rely heavily on the NLTK package to create our tokenized input. We use the `word_tokenize` to take convert each csv row into an array of tokens. These tokens can be individual words, punctation, or word pieces. We compute the frequency of each token in the csv file to create our vocabulary. Our vocabulary is composed on the 8000 most frequent tokens in the csv file. We restrict the vocabulary so that we limit the effects of the vanishing gradient problem. With too many tokens in our vocab, the softmax function divides the probability associated with the perdiction over too many vocabulary tokens. We remove stopwords, because they add no or minimal meaning to sentence, and as such can be neglected in creating our perdiction. We then add START and END tokens to each phrase to tell the model when each phrase begins and end. WE add UNK tokens to fill the place of words that are not in our vocabulary. We then use the Gensim package's Word2Vec class to create embeddings which correlate to each string.

## Forward Pass

We use the ExampleGenerator class in order to format our tokens such that they can be fed into the RNN model. It returns a sequence of tokens and the expected target output when called and maintains a pointer into the list of sentences so that it will not return duplicate values. The RNN model class takes a word dimension, hidden dimension, and the vocab -- the vocab is only used for predictions. The word dimensions and hidden dimensions, on the other hand, are used to format and initialize the weight matrices `U, W, and V`. In order to start the actual propagation, we call the RNN class which takes an example and breaks it into its target and example components, which are then fed to the model's `propagate_forward` method. This method iterates over each token in the example sequence and creates a predicted output at each time step. We pass dot product each input token with the first weight matrix U, and then add the hidden state of the previous time step dotted with the hidden weight matrix W to derived the next hidden state. The value of this computation is saved into the matStore property of the RNN model. It should be noted that the first timestep has no previous hidden state to use to perform this computation. We remedy this by adding the first time step to a matrix of zeros.