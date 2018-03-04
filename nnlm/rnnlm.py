import numpy as np
import keras
import argparse
import codecs
import matplotlib.pyplot as plot
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
# TBD: import necessary layers here
# from keras.layers import 

parser = argparse.ArgumentParser(description='Train unidirectional RNN')

parser.add_argument('--train-file', required=True,
                    help='File with sentence separated training data')
parser.add_argument('--dev-file', required=True,
                    help='File with sentence separated development data')
parser.add_argument('--max-length', default=25, type=int,
                    help='Maximum sequence length in tokens (Default: 25)')
parser.add_argument('--epochs', default=10, type=int,
                    help='Number of training epochs (Default: 10)')
parser.add_argument('--vocab-size', default=None, type=int,
                    help='Vocabulary size (Default: None)')

if __name__ == "__main__":
    # Dashes in argument names get replaced with underscores to create variable names
    args = parser.parse_args()
    seq_start = "@s@"
    seq_end = "@e@"
    max_tokens = args.max_length

    # Converting training and validation data into sequences
    trainf = codecs.open(args.train_file,'r','utf-8')
    traintexts = trainf.read().splitlines()
    devf = codecs.open(args.dev_file,'r','utf-8')
    devtexts = devf.read().splitlines()
    corpus = []
    for traintext in traintexts:
        corpus.append(seq_start+" "+traintext+" "+seq_end)
    for devtext in devtexts:
        corpus.append(seq_start+" "+devtext+" "+seq_end)
    tokenizer = Tokenizer(num_words=args.vocab_size,filters='!"#$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(corpus)
    #print(tokenizer.word_index)
    #print(tokenizer.word_counts)

    train_in = [seq_start+" "+x for x in traintexts]
    train_out = [x+" "+seq_end for x in traintexts]
    train_in_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(train_in),maxlen=max_tokens,padding='post',truncating='post')
    train_out_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(train_out),maxlen=max_tokens,padding='post',truncating='post')
    train_out_seq = np.expand_dims(train_out_seq,-1)

    dev_in = [seq_start+" "+x for x in devtexts]
    dev_out = [x+" "+seq_end for x in devtexts]
    dev_in_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(dev_in),maxlen=max_tokens,padding='post',truncating='post')
    dev_out_seq = sequence.pad_sequences(tokenizer.texts_to_sequences(dev_out),maxlen=max_tokens,padding='post',truncating='post')
    # https://github.com/keras-team/keras/issues/7303
    dev_out_seq = np.expand_dims(dev_out_seq,-1)
    vocab_size=len(tokenizer.word_counts)+1

    # Training the model
    embedding_dim = 32 
    rnn_dim = 16
    batch_size = 48

    model=Sequential()
    # TBD: Add model definition and compilation here

    model.summary()
    history = model.fit(train_in_seq,train_out_seq,
                        epochs = args.epochs,
                        batch_size = batch_size,
                        validation_data = (dev_in_seq,dev_out_seq))

    # Charting perplexity over the training epochs
    loss_values = history.history['loss']
    val_loss_values = history.history['val_loss']
    perp = np.exp2(history.history['loss'])
    val_perp = np.exp2(history.history['val_loss'])
    epochs = range(1, len(loss) + 1)

    plot.plot(epochs, loss_values, 'bo', label='Training loss')
    plot.plot(epochs, val_loss_values, 'b', label='Validation loss')
    plot.title('Training and validation loss')
    plot.xlabel('Epochs')
    plot.ylabel('Loss')
    plot.legend()
    plot.figure()

    plot.plot(epochs, perp, 'bo', label='Training perplexity')
    plot.plot(epochs, val_perp, 'b', label='Validation perplexity')
    plot.title('Training and validation perplexity')
    plot.xlabel('Epochs')
    plot.ylabel('Perplexity')
    plot.legend()
    plot.show()
