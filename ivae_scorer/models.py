import random

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import losses, regularizers
from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ivae_scorer.layers import Informed


def build_reactome_layers(adj, act="tanh"):
    return (
        Informed(
            adj=adj,
            name="pathways",
            activation=act,
            activity_regularizer=regularizers.L2(1e-5),
        ),
    )


def build_reactome_vae(adj):
    reactome_layers = build_reactome_layers(adj)
    return build_vae(layers=reactome_layers, seed=42, learning_rate=1e-5)


def build_vae(layers, seed, learning_rate):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    latent_dim = layers[-1].adj.shape[1] // 2
    input_dim = layers[0].adj.shape[0]

    inputs = Input(shape=(input_dim,))

    # build recursevely the hidden layers of the encoder
    for i, layer in enumerate(layers):
        if i == 0:
            inner_encoder = layer(inputs)
        else:
            inner_encoder = layer(inner_encoder)

    z_mean = Dense(latent_dim)(inner_encoder)
    z_log_sigma = Dense(latent_dim)(inner_encoder)

    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(
            shape=(K.shape(z_mean)[0], latent_dim), mean=0.0, stddev=0.1
        )
        return z_mean + K.exp(z_log_sigma) * epsilon

    z = Lambda(sampling)([z_mean, z_log_sigma])

    # Create encoder
    encoder = Model(inputs, [z_mean, z_log_sigma, z], name="encoder")

    # Create decoder
    latent_inputs = Input(shape=(latent_dim,), name="z_sampling")

    # build recursevely the hidden layers of the decoder
    for i, layer in enumerate(layers[::-1]):
        if i == 0:
            inner_decoder = Dense(layer.adj.shape[1], activation="tanh")(latent_inputs)
        else:
            inner_decoder = Dense(layer.adj.shape[1], activation="tanh")(inner_decoder)

    outputs = Dense(input_dim, activation="linear")(inner_decoder)
    decoder = Model(latent_inputs, outputs, name="decoder")

    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name="vae_mlp")

    reconstruction_loss = losses.mse(inputs, outputs)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=Adam(learning_rate=learning_rate), metrics=["mse"])

    return vae, encoder, decoder
