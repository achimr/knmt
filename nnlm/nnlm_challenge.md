Recurrent neural language model
-------------------------------

Please submit your neural network training code and a report answering
the questions below.

The task in this challenge is to implement the training of a recurrent
neural language model like described in section 4.4 of the [Koehn NMT
textbook](http://mt-class.org/jhu/assets/nmt-book.pdf) using the
[Keras](https://keras.io/) deep learning library.

### Required training time

While the challenge assignment does not require a lot of coding, training
neural language models on a desktop/laptop CPU can take several hours.
The assignment asks you to try different experiment configurations,
therefore the challenge needs some time to finish.

### Setting up the environment (10 points)

1.  Set up an Anaconda Python 3.5 environment as described
    in [Computing
    Environment](../install.md)
2.  Clone the knmt github repository <https://github.com/achimr/knmt> to
    your computer
3.  To verify that your setup is working
    1.  Launch an Anaconda py35 prompt
    2.  Change the current directory to the knmt/xor folder
    3.  Run \"python xor.py\" - the XOR network should be trained and a
        chart of the training loss displayed (include this chart in your
        report)

### Training a recurrent neural language model (30 points)

1.  The code template for the challenges below can be found in the file
    knmt/nnlm/rnnlm.py
2.  Extract the training and validation data nnlm\_data.zip 
    (compiled by Shuoyang Ding from Johns Hopkins University) 
    to the knmt/nnlm folder

#### Task

Your task is now to implement the training of a recurrent neural
language model like described in section 4.4 of the [Koehn NMT
textbook](http://mt-class.org/jhu/assets/nmt-book.pdf) using a word
embedding and a recurrent neural network layer. As in the feed forward
neural language model an output layer with the softmax activation
function needs to be present to ensure that the output of the network is
a proper probability distribution.

It is recommendable to start with the \"adam\"
[optimizer](https://keras.io/optimizers/) for this network and the
following.

The [loss function](https://keras.io/losses/) for this network and the
following needs to be crossentropy in order for the language model
perplexity to be calculated correctly for the charts (unfortunately we
have to rely on the calculation of the perplexity from the loss, as
calculating perplexity independently in Keras currently has a bug).

#### Data preparation in template code

The template code contains code to read training and validation files
that are specified via command line arguments:

``` {style="padding-left: 30px;"}
python rnnlm.py --train-file train.en.txt --dev-file dev.en.txt
```

(this will fail until you add the training code).

Training and validation data are converted by the template code into
integer sequences (with the numbers indicating vocabulary IDs) of the
shape

    (corpus length, maximum sequence length)

for the input and the shape

    (corpus lines, maximum sequence length, timestep)

for the output (both for training and validation data). The output
sequences are just the input sequences shifted left by one. Input
sentences shorter than the maximum sequence length are right-padded with
zeros (the Keras tokenizer does not assign zero to any tokens). Input
sequences longer than the maximum sequence length are truncated right.

While you are developing the networks and the modifications you might
want to use a subset of the training set of 1000 or 5000 sentences. Note
that this mainly serves debugging and development - you still should
verify modifications on the entire training set.

#### Baseline

Train a baseline RNNLM system (with the default parameters and provided
training and validation data) and save the resulting perplexity and loss
charts (the charting code is already contained in the template code).
The training will take about 2-3 hours on a Intel Core i5 laptop. What
do you observe about the training? Please include your observations and
the charts in the report.

#### Modification

Now make one modification and retrain the system. Possible modifications
are:

-   Expand (or reduce) the maximum sequence length - the default maximum
    sequence length is 25 which is the average sentence length of the
    training corpus in tokens. So a lot of the sentences will be
    truncated. Longer sequence lengths result in longer training times.
-   Reduce the vocabulary size (command line parameter \--vocab-size)
-   Add more recurrent layers (see deep stacked layers in Koehn NMT
    section 4.7)
-   Use a different optimizer
-   Use a different weight initialization
-   Measures to reduce over-fitting: adjust network size (e.g.
    rnn\_dim), weight normalization, dropout

What do you observe about the training with the modification? Does the
modified network perform better or worse than the baseline? Does the
modified network over-fit more or less to the training data? What have
you learned from the modification to improve the network? Try to
interpret your observations. You can suggest further steps to prove or
disprove your interpretations. Please report the training time for the
modified network. Please include your answers and charts in the report.

### Training a LSTM neural language model (30 points)

#### Task

In the network replace the recurrent neural network layer with a LSTM
layer and retrain the system.

#### Baseline

Train a baseline LSTMLM system (with the default parameters and provided
training and validation data) and save the resulting perplexity and loss
charts (the charting code is already contained in the template code).
What do you observe about the training? Please include your observations
and the charts in the report.

#### Modification

Now make one modification and retrain the system. Possible modifications
are:

-   Expand (or reduce) the maximum sequence length - the default maximum
    sequence length is 25 which is the average sentence length of the
    training corpus in tokens. So a lot of the sentences will be
    truncated. Longer sequence lengths result in longer training times.
-   Reduce the vocabulary size (command line parameter \--vocab-size)
-   Add more recurrent layers (see deep stacked layers in Koehn NMT
    section 4.7)
-   Use a different optimizer
-   Use a different weight initialization
-   Measures to reduce over-fitting: adjust network size (e.g.
    rnn\_dim), weight normalization, dropout

What do you observe about the training with the modification? Does the
modified network perform better or worse than the baseline? Does the
modified network over-fit more or less to the training data? What have
you learned from the modification to improve the network? Try to
interpret your observations. You can suggest further steps to prove or
disprove your interpretations. Please report the training time for the
modified network. Please include your answers and charts in the report.

You can make the same modification as for the RNNLM above or a different
one. Using the same modification allows you to make comparisons between
the different network architectures. Observations about these
comparisons should be included in the report.

### Training a bidirectional LSTM neural language model (30 points)

#### Task

In the network replace the LSTM layer with a bidirectional LSTM layer
and retrain the system. A bidirectional layer allows to learn from left
and right context.

#### Baseline

Train a baseline bidirectional LSTMLM system (with the default
parameters and provided training and validation data) and save the
resulting perplexity and loss charts (the charting code is already
contained in the template code). What do you observe about the training?
Please include your observations and the charts in the report.

#### Modification

Now make one modification and retrain the system. Possible modifications
are:

-   Expand (or reduce) the maximum sequence length - the default maximum
    sequence length is 25 which is the average sentence length of the
    training corpus in tokens. So a lot of the sentences will be
    truncated. Longer sequence lengths result in longer training times.
-   Reduce the vocabulary size (command line parameter \--vocab-size)
-   Add more recurrent layers (see deep stacked layers in Koehn NMT
    section 4.7)
-   Use a different optimizer
-   Use a different weight initialization
-   Measures to reduce over-fitting: adjust network size (e.g.
    rnn\_dim), weight normalization, dropout

What do you observe about the training with the modification? Does the
modified network perform better or worse than the baseline? Does the
modified network over-fit more or less to the training data? What have
you learned from the modification to improve the network? Try to
interpret your observations. You can suggest further steps to prove or
disprove your interpretations. Please report the training time for the
modified network. Please include your answers and charts in the report.

You can make the same modification as for the RNNLM and LSTMLM above or
a different one. Using the same modification allows you to make
comparisons between the different network architectures. Observations
about these comparisons should be included in the report.
