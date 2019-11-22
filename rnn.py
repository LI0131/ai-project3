import os
import logging
import tensorflow as tf

from keras.preprocessing import sequence
from keras.datasets import imdb

logging.basicConfig(filename='info.log', level=logging.INFO)

DROP_OUT_RATE = os.environ.get('DROP_OUT_RATE', 0.3)
OPTIMIZER = os.environ.get('OPTIMIZER', 'adam')
LOSS_FUNCTION = os.environ.get('LOSS_FUNCTION', 'binary_crossentropy')
BATCH_SIZE = os.environ.get('BATCH_SIZE', 64)
VOCAB_SIZE = os.environ.get('VOCAB_SIZE', 20000)


def run(bidirectional=False, deepdeep=False):
    logging.info('Pulling data from Keras')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=VOCAB_SIZE)

    x_train = sequence.pad_sequences(x_train)
    x_test = sequence.pad_sequences(x_test)

    logging.info('Init Model')
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Embedding(VOCAB_SIZE, 64))
    if bidirectional:
        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(64, dropout=DROP_OUT_RATE, return_sequences=True)
            )
        )
    else:
        model.add(tf.keras.layers.LSTM(64, dropout=DROP_OUT_RATE, return_sequences=True))
    if deepdeep:
        model.add(tf.keras.layers.LSTM(64, dropout=DROP_OUT_RATE))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        loss=LOSS_FUNCTION,
        optimizer=OPTIMIZER,
        metrics=['accuracy']
    )

    logging.info('Begin Training')
    model.fit(
        x_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=3,
        validation_data=(x_test, y_test)
    )

    score, acc = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)

    logging.info(f'Test score: {score}')
    logging.info(f'Test accuracy: {acc}')


if __name__ == '__main__':
    logging.info('STARTING Standard RNN...')
    run()
    
    logging.info('STARTING Bidirectional RNN...')
    run(bidirectional=True)

    logging.info('STARTING Deep Deep RNN...')
    run(deepdeep=True)

    logging.info('STARTING Bidirectional Deep Deep RNN...')
    run(bidirectional=True, deepdeep=True)
