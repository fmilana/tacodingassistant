import io
import itertools
import numpy as np
import os
import re
import string
import tensorflow as tf
import tqdm
import datetime
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import (
    Activation,
    Dense,
    Dot,
    Embedding,
    Flatten,
    GlobalAveragePooling1D,
    Reshape
)
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorboard.plugins import projector


SEED = 42
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Read the txt file and create a TextLineDataset with non-empty lines.
path_to_file = 'text/joint_groupbuy_jhim.txt'
text_ds = tf.data.TextLineDataset(path_to_file).filter(
    lambda x: tf.cast(tf.strings.length(x), bool))


def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  return tf.strings.regex_replace(lowercase,
                                  '[%s]' % re.escape(string.punctuation), '')


# Define the vocabulary size and number of words in a sequence.
file = open(path_to_file, "rt")
vocab_size = 26847
sequence_length = 10

# Use a text vectorization layer to normalize, split, and map strings to
# integers. Set output_sequence_length length to pad all samples to same length.
vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=vocab_size,
    output_mode='int',
    output_sequence_length=sequence_length)
vectorize_layer.adapt(text_ds.batch(1024))
# Save the created vocabulary for reference.
inverse_vocab = vectorize_layer.get_vocabulary()

# Vectorize the data in text_ds.
text_vector_ds = text_ds.batch(1024).prefetch(AUTOTUNE).map(
    vectorize_layer).unbatch()

# Flatten the dataset into a list of sentence vector sequences.
sequences = list(text_vector_ds.as_numpy_iterator())

# Set the number of negative samples per positive context word.
num_ns = 4

# Generates lists of targets, contexts and labels to be used as training data.
def generate_training_data(sequences, window_size, num_ns, vocab_size, seed):
    # Elements of each training example are appended to these lists.
    targets, contexts, labels = [], [], []
    # Build the sampling table for vocab_size tokens.
    sampling_table = tf.keras.preprocessing.sequence.make_sampling_table(
        vocab_size)
    # Iterate over all sequences (sentences) in dataset.
    for sequence in tqdm.tqdm(sequences):
        # Generate positive skip-gram pairs for the sequence (sentence).
        positive_skip_grams, _ = tf.keras.preprocessing.sequence.skipgrams(
            sequence,
            vocabulary_size=vocab_size,
            sampling_table=sampling_table,
            window_size=window_size,
            negative_samples=0)
        # Iterate over each positive skip-gram pair to produce training
        # examples with a positive context word and negative samples.
        for target_word, context_word in positive_skip_grams:
            context_class = tf.expand_dims(tf.constant(
                [context_word], dtype='int64'), 1)
            negative_sampling_candidates, _, _ = tf.random.log_uniform_candidate_sampler(
                    true_classes=context_class,
                    num_true=1,
                    num_sampled=num_ns,
                    unique=True,
                    range_max=vocab_size,
                    seed=SEED,
                    name='negative_sampling')
            # Build context and label vectors (for one target word)
            negative_sampling_candidates = tf.expand_dims(
                negative_sampling_candidates, 1)
            context = tf.concat(
                [context_class, negative_sampling_candidates], 0)
            label = tf.constant([1] + [0]*num_ns, dtype='int64')
            # Append each element from the training example to global lists.
            targets.append(target_word)
            contexts.append(context)
            labels.append(label)
    # Return lists of targets, contexts and labels.
    return targets, contexts, labels


# Generate training examples from sequences.
targets, contexts, labels = generate_training_data(
    sequences=sequences,
    window_size=2,
    num_ns=4,
    vocab_size=vocab_size,
    seed=SEED)

# Create Dataset object of (target_word, context_word), (label) elements
# to train the Word2Vec model.
BATCH_SIZE = 10
BUFFER_SIZE = 10000
dataset = tf.data.Dataset.from_tensor_slices(((targets, contexts), labels))
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
dataset = dataset.cache().prefetch(buffer_size=AUTOTUNE)


# Classifier to distinguish between true context words from skip-grams
# and false context words obtained through negative sampling.
class Word2Vec(Model):
    def __init__(self, vocab_size, embedding_dim):
        super(Word2Vec, self).__init__()
        self.target_embedding = Embedding(
            vocab_size,
            embedding_dim,
            input_length=1,
            name='w2v_embedding')
        self.context_embedding = Embedding(
            vocab_size,
            embedding_dim,
            input_length=num_ns+1)
        self.dots = Dot(axes=(3, 2))
        self.flatten = Flatten()

    def call(self, pair):
        target, context = pair
        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])
        return self.flatten(dots)


# Build the model.
embedding_dim = 128
word2vec = Word2Vec(vocab_size, embedding_dim)
word2vec.compile(
    optimizer='adam',
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])


# Callback to log training statistics for tensorboard.
timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = 'logs\\' + timestamp
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

# Train the model with dataset prepared above for some number of epochs.
word2vec.fit(dataset, epochs=20, callbacks=[tensorboard_callback])
# ----------------------------------------------------------------------


weights = word2vec.get_layer('w2v_embedding').get_weights()[0]
vocab = vectorize_layer.get_vocabulary()
# Create and save the metadata and words_vectors file.
with open(os.path.join(log_dir, 'metadata.tsv'), 'w') as fm:
    with open(os.path.join(log_dir, 'words_vectors.tsv'), 'w') as fwv:
        for index, word in enumerate(vocab):
            if index == 0: continue
            fm.write('{}\n'.format(word))
            vec = weights[index]
            fwv.write(word + '\t' + str([x for x in vec]) + '\n')

# Create and save the vectors file.
with open(os.path.join(log_dir, 'vectors.tsv'), 'w') as f:
    for index in range(1, len(vocab)):
        vec = weights[index]
        f.write('\t'.join([str(x) for x in vec]) + '\n')

# Create a checkpoint from embedding, the filename and key are
# name of the tensor.
checkpoint_weights = tf.Variable(word2vec.layers[0].get_weights()[0][1:])
# Create a checkpoint from embedding, the filename and key are
# name of the tensor.
checkpoint = tf.train.Checkpoint(embedding=checkpoint_weights)
checkpoint.save(os.path.join(log_dir, 'embedding.ckpt'))

# Set up config.
config = projector.ProjectorConfig()
embedding = config.embeddings.add()
# The name of the tensor will be suffixed by `/.ATTRIBUTES/VARIABLE_VALUE`
embedding.tensor_name = 'embedding/.ATTRIBUTES/VARIABLE_VALUE'
embedding.metadata_path = timestamp + '/metadata.tsv'
projector.visualize_embeddings(log_dir, config)
# ----------------------------------------------------------------------
