'''Restore a word-level sequence to sequence model from disk and use it
to generate predictions.

Adapted from https://github.com/keras-team/keras/tree/master/examples

This script loads the s2sw.h5 model saved by lstm_seq2seq_wordbased.py and generates
sequences from it.  It assumes that no changes have been made (for example:
latent_dim is unchanged, and the input data and model architecture are unchanged).

See lstm_seq2seq_wordbased.py for more details on the model architecture and how
it is trained.
'''
from __future__ import print_function

from keras.models import Model, load_model
from keras.layers import Input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
import numpy as np
import argparse
from nltk.translate.bleu_score import corpus_bleu

parser = argparse.ArgumentParser(description='Sequence-to-sequence NMT')

parser.add_argument('--train-file', default='fra-eng/fr_en.train.txt',
                    help='File with tab-separated parallel training data')
parser.add_argument('--test-file', default='fra-eng/fr_en.test_small.txt',
                    help='File with tab-separated parallel test data')
parser.add_argument('--num-samples', default=10000, type=int,
                    help='Number of samples to train on (Default: 10000)')
args = parser.parse_args()

latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = args.num_samples  # Number of samples to train on.

# Vectorize the data.  We use the same approach as the training script.
# NOTE: the data must be identical, in order for the character -> integer
# mappings to be consistent.
# We omit encoding target_texts since they are not needed.
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

# Restore the model and construct the encoder and decoder.
model = load_model('s2sw.h5')
# HERE
encoder_inputs = model.input[0]   # input_1
encoder_outputs, state_h, state_c = model.layers[4].output   # lstm_1
encoder_states = [state_h, state_c]
encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = model.input[1]   # input_2
decoder_state_input_h = Input(shape=(latent_dim,), name='input_3')
decoder_state_input_c = Input(shape=(latent_dim,), name='input_4')
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = model.layers[5] # lstm_2
decoder_embed = model.layers[3].output # embed_2
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_embed, initial_state=decoder_states_inputs)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = model.layers[6]
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


# Decodes an input sequence.  Future work should support beam search.
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
