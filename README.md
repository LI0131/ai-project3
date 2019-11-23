# ai-project3
Recurrent Neural Network for Natural Language Processing

# Dev Setup

### Run Tokenization
`python3 app.py`

### Run RNN
`python3 rnn.py`

# About This Project

## Tokenization

Tokenization is the process of transforming an input file into a format which is acceptable for creating embeddings. We rely heavily on the NLTK package to create our tokenized input. We use the `word_tokenize` to take convert each csv row into an array of tokens. These tokens can be individual words, punctation, or word pieces. We compute the frequency of each token in the csv file to create our vocabulary. Our vocabulary is composed on the 8000 most frequent tokens in the csv file. We restrict the vocabulary so that we limit the effects of the vanishing gradient problem. With too many tokens in our vocab, the softmax function divides the probability associated with the perdiction over too many vocabulary tokens. We remove stopwords, because they add no or minimal meaning to sentence, and as such can be neglected in creating our perdiction. We then add START and END tokens to each phrase to tell the model when each phrase begins and end. WE add UNK tokens to fill the place of words that are not in our vocabulary. We then use the Gensim package's Word2Vec class to create embeddings which correlate to each string.

## Forward Pass

We use the ExampleGenerator class in order to format our tokens such that they can be fed into the RNN model. It returns a sequence of tokens and the expected target output when called and maintains a pointer into the list of sentences so that it will not return duplicate values. The RNN model class takes a word dimension, hidden dimension, and the vocab -- the vocab is only used for predictions. The word dimensions and hidden dimensions, on the other hand, are used to format and initialize the weight matrices `U, W, and V`. In order to start the actual propagation, we call the RNN class which takes an example and breaks it into its target and example components, which are then fed to the model's `propagate_forward` method. This method iterates over each token in the example sequence and creates a predicted output at each time step. We pass dot product each input token with the first weight matrix U, and then add the hidden state of the previous time step dotted with the hidden weight matrix W to derived the next hidden state. The value of this computation is saved into the matStore property of the RNN model. It should be noted that the first timestep has no previous hidden state to use to perform this computation. We remedy this by adding the first time step to a matrix of zeros.

## Full RNN Implementation using Keras and Tensorflow
  
In order to obtain the data, we imported it from Keras using the predefined breakup of training and testing data. As a parameter, Keras allows for us to pass the vocal size, and it will automatically limit the vocabulary. We chose a vocal size of 20,000 so that we are able to read and calculate nearly all words in the dataset, without having an infinite vocal size. We padded the data to match the largest piece of data. This allows for us to concatenate data pieces and perform operations simultaneously in batch sizes of 64 to speed up training. We initialized the model. For each of our four models, we used a dropout rate of 30%, binary cross entropy as a loss function, and the adam optimizer. 

The first model we tested was a standard LSTM. We suspected that this model would have the lowest accuracy. It's accuracy against the testing set capped out at roughly 87%.

This accuracy was obtained without using truncation during data preprocessing to speed up the training process. So the input was padded to the size of the largest input item in the set. We truncated the size of the model to size 80 when testing with the more complex models. This allowed us to more rapidly test how performant each model was to the others.

The second model we tested was a bidirectional LSTM. We suspected that this model would have higher accuracy that the standard LSTM. This is because a bidirectional LSTM is able to value both the forward and backward dependencies of tokens by passing through the sequence both forwards and backwards. Its accuracy value was measured on the truncated data set as 82.35% accurate which we can compare to 82.5% accuracy for the standard LSTM on the truncated set.

The third model we tested was a deep deep standard LSTM. Adding an additional layer allows the model to extrapolate more complex hierarchical features from the data, thus providing a more accurate prediction. We use the `return_sequences` parameter in order to make this functionality work, since the inclusion of this parameter fits the shape of the output data of the first layer to appropriate fit the second layer. The accuracy of this model on the truncated set was 83.3%.

The fourth model we tested was a deep deep bidirectional LSTM. We suspected that this would be the most accurate model. The deep deep nature allows the model to extrapolate more complex features, and the bidirectionally allows the model to form both forward and backward dependencies. We found this model to actually not outperform the deep deep model. On the truncated set it only achieved an accurate measure of 82.54%.

The most important thing to note here is that the accuracies of each model were relatively similar. While training more complex models can help us achieve a higher accuracy, it is still relatively low. This is because the data that we are modeling is not in nature a perfect fit for an RNN. While is it sequential data, sentiment analysis does not have as many long range dependencies as other sequential data modeling. In fact, modeling sentiment analysis with an RNN may actually extrapolate long range dependencies that are not there, thus lowering accuracy. In this case, a CNN may be a better model to use. 

Furthermore, Training with more complex models also leads to longer training times for what can be marginal improvements in accuracy. While we are able to obtain more accurate predictions using a deep deep bidirectional RNN, it takes much more time to train than a traditional LSTM model. It may make sense then, depending on the application of the model to use a less complex model with more efficient training times.

### Sources ###
https://github.com/keras-team/keras/blob/master/examples/imdb_lstm.py
