'''
Summary of the algorithm

We start with input sequences from a domain (e.g. English sentences) and corresponding target sequences from another domain (e.g. French sentences).
An encoder LSTM turns input sequences to 2 state vectors (we keep the last LSTM state and discard the outputs).
A decoder LSTM is trained to turn the target sequences into the same sequence but offset by one timestep in the future, a training process called "teacher forcing" in this context. It uses as initial state the state vectors from the encoder. Effectively, the decoder learns to generate targets[t+1...] given targets[...t], conditioned on the input sequence.
In inference mode, when we want to decode unknown input sequences, we:
Encode the input sequence into state vectors
Start with a target sequence of size 1 (just the start-of-sequence character)
Feed the state vectors and 1-char target sequence to the decoder to produce predictions for the next character
Sample the next character using these predictions (we simply use argmax).
Append the sampled character to the target sequence
Repeat until we generate the end-of-sequence character or we hit the character limit.
'''


from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np

batch_size = 64  # Batch size for training.
epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
# Path to the data txt file on disk.
data_path = 'fra.txt'

# Vtorize the data.
input_texts = []
target_texts = []
input_characters = set()
target_characters = set()
with open(data_path, 'r', encoding='utf-8') as f:
    #read the input as lines
    lines = f.read().split('\n')
#read a maximum of 10000 lines and make it as input
for line in lines[: min(num_samples, len(lines) - 1)]:
#input =english sentence, target = french sentence
    input_text, target_text, _ = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.
    target_text = '\t' + target_text + '\n'
    input_texts.append(input_text)
    target_texts.append(target_text)
    #create vocabulary of unqiue input characters
    for char in input_text:
        if char not in input_characters:
            input_characters.add(char)
    #create vocabulary of unquie target characters
    for char in target_text:
        if char not in target_characters:
            target_characters.add(char)
#create a sorted list of input and target characters
input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
#total no of encode and decoder tokens
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
#maximun input sentece length and target sentence length
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)
#build a char to index look up for input characters
input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
#build a char to index look up for target characters
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])
#no of lines*no of characters*vector rpresenting all unquie input characters
encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
#no of lines*no of characters*vector rpresenting all unquie output characters
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
#create the target output matrix like the input matrix
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
#onehot encoding of input and target data
for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    #encode space as the last dimension in the onehot vector
    encoder_input_data[i, t + 1:, input_token_index[' ']] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.
    decoder_input_data[i, t + 1:, target_token_index[' ']] = 1.
    decoder_target_data[i, t:, target_token_index[' ']] = 1.
# Define an input sequence and process it.
#number unquie charecters in input data will be passed to the lstm
encoder_inputs = Input(shape=(None, num_encoder_tokens))
#generate a 256 dimension space embedding foe each character
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
#state_h= LSTM hidden state, state_c = LSTM cell state
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
#number of unquie target characters for the ouputt sequence
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
#generate a latent dimension vector decoder sequences
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
#intializing decoder lstm to generate output sequence lstm cell output = no of unquie decoder input characters
#pass the encoder intial state to decoder
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
#number of possible classes
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
#pass the ouput from the nlstm to the dense layer
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

# Run training
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)
# Save model
model.save('s2s.h5')

# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)
#intialise decoder states
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
#predict the decoder output based on previous state
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
#these state will represent the first character
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
#append the first letter to any previously exsisting state of the decoder_lstm
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        #get predictions for the target by passing output from the previous timestwp, states from the encoder_model
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        #get the character with the highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence


for seq_index in range(100):
    # Take one sequence (part of the training set)
    # for trying out decoding.
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)
