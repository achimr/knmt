Neural machine translation with the sequence-to-sequence model
--------------------------------------------------------------

In this assignment I provide an implementation of the
sequence-to-sequence model for neural machine translation like described
in section 5.1 of the [Koehn NMT
textbook](http://mt-class.org/jhu/assets/nmt-book.pdf) and Sutskever et.
al. [Sequence to Sequence Learning for Neural
Networks](https://arxiv.org/abs/1409.3215), 2014. It is also described
in the Tensorflow [Neural Machine Translation (seq2seq)
Tutorial](https://www.tensorflow.org/tutorials/seq2seq). Your task will
be to train a model and evaluate it with the BLEU score. Next you will
improve on the quality of the model by changing the training
configuration and by improving the algorithm.

Please submit a report answering the analysis questions below and the
modified code.

### Required training time

While the challenge assignment does not require a lot of coding, training
neural MT models on a desktop/laptop CPU can take several hours. The
assignment asks you to try different experiment configurations,
therefore the challenge needs to be started early to allow for the
different experiments to finish.

### Training the model and evaluating the output with BLEU (20 points)

1.  Set up an Anaconda Python 3.5 environment as described
    in [Computing
    Environment](../install.md)
2.  Clone the knmt github repository <https://github.com/achimr/knmt> to
    your computer 
3.  Open and Anaconda 3.5 prompt and navigate to the knmt/seq2seq folder
4.  Extract the training and test data from `seq2seq_data.zip` and `seq2seq_train_shuffled.zip` to a folder called fra-eng below the knmt/seq2seq folder
5.  Run \"python lstm\_seq2seq\_wordbased.py\" to train a baseline NMT
    system. This will\
    -   Train a French-English seq2seq NMT model on the first 8000
        sentence pairs in fr-en.train.txt (for 100 epochs taking about 2
        hours on a Core i5 laptop)
    -   Write the trained model to a file s2sw.h5
    -   Run inference on the source contained in fr-en.test\_small.txt
    -   Calculate the BLEU score on the inference output with the
        reference sentences contained in fr-en.test\_small.txt - the
        baseline BLEU score should be around 0.03179288690658304

CAUTION: If you get a warning message like the following the displayed
BLEU score will not be valid and it is not comparable to other BLEU
scores calculated without the warning. You should treat it as a BLEU
score of zero.

    C:\Users\achim\Anaconda3\envs\py35\lib\site-packages\nltk\translate\bleu_score.py:490: UserWarning:
    Corpus/Sentence contains 0 counts of 3-gram overlaps.
    BLEU scores might be undesirable; use SmoothingFunction().
      warnings.warn(_msg)
    BLEU score: 0.09686693290317193

 

Make sure you copy and paste the output into a text file. In your report
provide an analysis on:

1.  The characteristics of the training data that is used for the
    baseline
2.  The progression of the training and validation data loss during the
    training
3.  The translations of the test set: Which ones are good? Which ones
    bad? What is missing? What is characteristic about the source
    sentences that have bad translations? (you should not do a
    sentence-by-sentence analysis, but rather high-level observations)
    What are potential reasons for the bad translations/missing
    information?
4.  The BLEU score

Once you have trained the model, you can read the trained model from the
file s2sw.h5 and run inference using the script
lstm\_seq2seq\_restore\_wordbased.py. Run \"python
lstm\_seq2seq\_restore\_wordbased.py\" at least once without any changes
and verify that you get the same translations and same BLEU score as
with the full training.

### Improving the MT quality without changing the code (40 points)

Try to improve the BLEU score on the fr-en.test\_small.txt test set by
retraining the system with a different configuration. Possibilities are
(in no particular order):

-   Training the system for more (or less) epochs (command line
    parameter \--epochs)
-   Training the system with more training data (command line parameter
    \--num-samples). Note that the training data is structured in a way
    that the sentences get longer, so training time will increase
    because of this in addition to the increased training time because
    of the increased number of training sentence pairs.
-   Training with a different Embedding dimension (no command line
    parameter, variable latent\_dim)
-   Training with a different LSTM layer dimension (no command line
    parameter, variable latent\_dim)
-   Lowercase the training data
-   Pre-process the training data with a different tokenizer

Please analyze the effect of your changes on the translated sentences
and the BLEU score. Analyze why your changes improved the BLEU score or
not. It is sufficient to make one change that improves the BLEU score.
Please provide a log of your output.

### Improving the MT quality by improving the algorithm (40 points)

Try to improve the BLEU score on the fr-en.test\_small.txt test set by
retraining the system with a code change. Possibilities are (in no
particular order):

-   Add a Dropout layer to avoid over-fitting to the training data
-   Add additional LSTM layers (e.g. like described in the [Tensorflow
    seq2seq tutorial](https://www.tensorflow.org/tutorials/seq2seq))
-   Reverse the input sentence like described in Sutskever et. al. 2014
-   Implement attention like described in Bahdanau, Cho & Bengio,
    [Neural Machine Translation by Jointly Learning to Align and
    Translate](https://arxiv.org/abs/1409.0473),
    ICLR 2015 or refined in Luong et. al., [Effective Approaches to
    Attention-based Neural Machine
    Translation](https://arxiv.org/abs/1508.04025), 2015. A library to
    implement attention in Keras can be found in this
    [tutorial](https://medium.com/datalogue/attention-in-keras-1892773a4f22).
-   Implement beam decoding like described in section 5.4 of the Koehn
    NMT book
-   Add an attention model

Please analyze the effect of your changes on the translated sentences
and the BLEU score. Analyze why your changes improved the BLEU score or
not. It is sufficient to make one change that improves the BLEU score
over the BLEU score from the previous task (Improving the MT quality
without changing the code). Please provide a log of your output and your
modified code.

Of course the options above require different levels of work, but I will
not compare one solution to the other in grading. Important for grading
is a code change leading to an output quality improvement and the
analysis why it did.

#### Acknowledgments

-   The word-based seq2seq code was adapted from a character-based
    seq2seq model in the Keras examples
    <https://github.com/keras-team/keras/tree/master/examples>
-   The parallel data was adapted from Tatoeba data by the maintainers
    of and is licensed under a [Creative Commons Attribution License 2.0
    (fr)](https://creativecommons.org/licenses/by/2.0/fr/)
