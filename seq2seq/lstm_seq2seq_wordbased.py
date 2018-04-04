'''Sequence to sequence example in Keras (word-level).

Adapted from https://github.com/keras-team/keras/tree/master/examples

This script demonstrates how to implement a basic word-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Word-level machine translation models are common.

# Summary of the algorithm

- We start with input sequences from a domain (e.g. English sentences)
    and correspding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence token)
    - Feed the state vectors and 1-token target sequence
        to the decoder to produce predictions for the next token
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled word to the target sequence
    - Repeat until we generate the end-of-sequence token or we
        hit the word limit.

# Data download

English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function

from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.layers import Input, LSTM, Dense, Embedding
import numpy as np
import argparse
from nltk.translate.bleu_score import corpus_bleu

parser = argparse.ArgumentParser(description='Sequence-to-sequence NMT')

parser.add_argument('--train-file', default='fra-eng/fr_en.train.txt',
                    help='File with tab-separated parallel training data')
parser.add_argument('--test-file', default='fra-eng/fr_en.test_small.txt',
                    help='File with tab-separated parallel test data')
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of training epochs (Default: 100)')
parser.add_argument('--num-samples', default=10000, type=int,
                    help='Number of samples to train on (Default: 10000)')
args = parser.parse_args()

batch_size = 64  # Batch size for training.
epochs = args.epochs  # Number of epochs to train for.
embedding_dim = 256 # Dimensionality of the word embedding
latent_dim = 256  # Latent dimensionality of LSTM layer
num_samples = args.num_samples  # Number of samples to train on.

# Vectorize the data.
input_texts = []
target_texts = []
seq_start = "@s@"
seq_end = "@e@"

input_characters = set()
target_characters = set()
input_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n',oov_token='@o@')
target_tokenizer = Tokenizer(filters='!"#$%&()*+,-./:;<=>?[\\]^_`{|}~\t\n',oov_token='@o@')

with open(args.train_file, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    target_text = seq_start+' ' + target_text + ' ' + seq_end
    input_texts.append(input_text)
    target_texts.append(target_text)
input_tokenizer.fit_on_texts(input_texts)
target_tokenizer.fit_on_texts(target_texts)

# Adding two to number of tokens: 0 and index for OOV
num_encoder_tokens = len(input_tokenizer.word_index)+2
num_decoder_tokens = len(target_tokenizer.word_index)+2
encoder_input_seq = input_tokenizer.texts_to_sequences(input_texts)
decoder_input_seq = target_tokenizer.texts_to_sequences(target_texts)
max_encoder_seq_length = max([len(txt) for txt in encoder_input_seq])
max_decoder_seq_length = max([len(txt) for txt in decoder_input_seq])
#decoder_outputs = [a[1:]+a[:1] for a in decoder_inputs]
decoder_output_seq = [a[1:]+[0] for a in decoder_input_seq]
reverse_input_word_index = dict(
    (i, word) for word, i in input_tokenizer.word_index.items())
reverse_target_word_index = dict(
    (i, word) for word, i in target_tokenizer.word_index.items())

encoder_input_data = sequence.pad_sequences(encoder_input_seq,maxlen=max_encoder_seq_length,padding='post',truncating='post')
decoder_input_data = sequence.pad_sequences(decoder_input_seq,maxlen=max_decoder_seq_length,padding='post',truncating='post')
decoder_output_data = sequence.pad_sequences(decoder_output_seq,maxlen=max_decoder_seq_length,padding='post',truncating='post')
decoder_output_data = np.expand_dims(decoder_output_data,-1)

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None,))
x = Embedding(num_encoder_tokens, embedding_dim)(encoder_inputs)
x, state_h, state_c = LSTM(latent_dim,
                           return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None,))
decoder_embed = Embedding(num_decoder_tokens, embedding_dim)
x = decoder_embed(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True,return_state=True)
decoder_outputs, _, _ = decoder_lstm(x, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_inputs` & `decoder_inputs` into `decoder_outputs`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

#import pdb; pdb.set_trace()
# Compile & run training
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
# Using sparse_categorical_crossentropy to allow for use of integer indexes
# for `decoder_target_data`
model.fit([encoder_input_data, decoder_input_data], decoder_output_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)

# Save model
model.save('s2sw.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states


# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm_inputs = decoder_embed(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_lstm_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_word_index = dict(
    (i, word) for word, i in input_tokenizer.word_index.items())
reverse_input_word_index[0] = ''
reverse_target_word_index = dict(
    (i, word) for word, i in target_tokenizer.word_index.items())
reverse_target_word_index[0] = ''

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    # Populate the first word of target sequence with the start token
    target_seq = np.array([target_tokenizer.word_index[seq_start]])

    #import pdb; pdb.set_trace()
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #import pdb; pdb.set_trace()
        sampled_word = reverse_target_word_index[sampled_token_index]
        if(sampled_word != seq_end):
            decoded_sentence += [sampled_word]

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_word == seq_end or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.array([sampled_token_index])

        # Update states
        states_value = [h, c]

    return decoded_sentence

input_test_texts = []
ref_test_texts = []
with open(args.test_file, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[:len(lines)-1]:
    input_test_text, ref_test_text = line.split('\t')
    input_test_texts.append(input_test_text)
    ref_test_texts.append(ref_test_text)
input_test_seqs = input_tokenizer.texts_to_sequences(input_test_texts)
input_test_data = sequence.pad_sequences(input_test_seqs,maxlen=max_encoder_seq_length,padding='post',truncating='post')
ref_test_seqs = target_tokenizer.texts_to_sequences(ref_test_texts)
decoded = []
references = []
#import pdb; pdb.set_trace()
for input_data,input_text in zip(input_test_data,input_test_texts):
    input_seq = np.expand_dims(input_data,0)
    decoded_sentence_array = decode_sequence(input_seq)
    decoded_sentence = " ".join(decoded_sentence_array)
    decoded.append(decoded_sentence_array)
    print('-')
    print('Input sentence:', input_text)
    print('Decoded sentence:', decoded_sentence)
for ref_seq in ref_test_seqs:
    reference_array = [reverse_target_word_index[i] for i in ref_seq]
    references.append([reference_array])
#import pdb; pdb.set_trace()
bleu_score = corpus_bleu(references,decoded)
print('BLEU score:', bleu_score)
