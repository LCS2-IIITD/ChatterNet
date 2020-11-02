text_embedding_dim = 100
from keras import layers
from keras import backend as K
import tensorflow as tf
from keras.models import Model
from adaptive_cnn import adaptive_conv
from text_cnn import text_cnn
import numpy as np
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects


def exp_act(x):
    return K.exp

get_custom_objects().update({'exp_act': Activation(exp_act)})


emb_weights = np.load('../model_codes/embedding_weights.npy')
def masked_mse(mask_value):
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = K.square(mask_true * (y_true - y_pred))
        masked_mse = K.sum(masked_squared_error, axis=-1) / (K.sum(mask_true, axis=-1)+1.)
        return masked_mse
    #f.__name__ = 'Masked MSE (mask_value={})'.format(mask_value)
    return f
    
def masked_relative_error(mask_value):
    def f(y_true, y_pred):
        mask_true = K.cast(K.not_equal(y_true, mask_value), K.floatx())
        masked_squared_error = mask_true * (K.abs(y_true - y_pred)/(y_true+K.epsilon()))
        error = K.sum(masked_squared_error, axis=-1) / (K.sum(mask_true, axis=-1)+1.)
        return error
    #f.__name__ = 'Masked MSE (mask_value={})'.format(mask_value)
    return f
    

def build_model(news_per_hour,
                token_per_news,
                sub_per_hour,
                token_per_sub):

    news_input = layers.Input(batch_shape=(1,
                                           news_per_hour,
                                           token_per_news))
    sub_input = layers.Input(batch_shape=(1,
                                          sub_per_hour,
                                          token_per_sub))
    sub_predict = layers.Input(batch_shape=(1,
                                            sub_per_hour,
                                            token_per_sub))
                                            
    comment_rate = layers.Input(batch_shape=(1,
                                             sub_per_hour,
                                             1))
                                            
    subred_input = layers.Input(batch_shape=(1,
                                            sub_per_hour,))                                        
    subred_pred = layers.Input(batch_shape=(1,
                                           sub_per_hour,))
                                                                                                                            
    embedding_layer = layers.Embedding(input_dim=emb_weights.shape[0],
                                       output_dim=100,
                                       weights = [emb_weights],
                                       trainable=True
                                       )
    subreddit_embedding = layers.Embedding(input_dim=44, output_dim=32)
    
    
    news_text = embedding_layer(news_input)
    sub_text = embedding_layer(sub_input)
    predict_text = embedding_layer(sub_predict)
    
    subred_input_embedded = subreddit_embedding(subred_input)
    subred_pred_embedded = subreddit_embedding(subred_pred)
    
    
    #sub_pred = layers.Concatenate(axis=-1)([sub_pred, subred_pred_embedded])
    

    N = layers.TimeDistributed(text_cnn(token_per_news,
                                        text_embedding_dim))(news_text)
    S = layers.TimeDistributed(text_cnn(token_per_sub,
                                        text_embedding_dim))(sub_text)
    S = layers.Concatenate(axis=-1)([S, subred_input_embedded])
    
    news_state = layers.GRU(128,
                                 #return_state=True,
                                 stateful=True)(N)
    sub_state = layers.GRU(128,
                                #return_state=True,
                                stateful=True)(S)
    state = layers.Concatenate(axis=-1)([news_state, sub_state])

    adaconv5_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=5, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool5_1 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_1)
    adaconv5_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=5, rank=1, padding='same', activation='relu')([pool5_1, state, subred_pred_embedded])
    pool5_2 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_2)
    adaconv5_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=5, rank=1, padding='same', activation='relu')([pool5_2, state, subred_pred_embedded])
    pool5_3 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_3)
    #flat5 = layers.Flatten()(pool5_3)
    shape = K.int_shape(pool5_3)
    rep5 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool5_3)
    
    adaconv3_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=3, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool3_1 = layers.TimeDistributed(layers.MaxPool1D(3, padding='same'))(adaconv3_1)
    adaconv3_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=3, rank=1, padding='same', activation='relu')([pool3_1, state, subred_pred_embedded])
    pool3_2 = layers.TimeDistributed(layers.MaxPool1D(3, padding='same'))(adaconv3_2)
    adaconv3_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=4, rank=1, padding='same', activation='relu')([pool3_2, state, subred_pred_embedded])
    pool3_3 = layers.TimeDistributed(layers.MaxPool1D(13, padding='same'))(adaconv3_3)
    #flat3 = layers.Flatten()(pool3_3)
    shape = K.int_shape(pool3_3)
    rep3 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool3_3)

    adaconv1_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=1, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool1_1 = layers.TimeDistributed(layers.MaxPool1D(1, padding='same'))(adaconv1_1)
    adaconv1_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=1, rank=1, padding='same', activation='relu')([pool1_1, state, subred_pred_embedded])
    pool1_2 = layers.TimeDistributed(layers.MaxPool1D(1, padding='same'))(adaconv1_2)
    adaconv1_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=1, rank=1, padding='same', activation='relu')([pool1_2, state, subred_pred_embedded])
    pool1_3 = layers.TimeDistributed(layers.MaxPool1D(150, padding='same'))(adaconv1_3)
    #flat1 = layers.Flatten()(pool1_3)
    shape = K.int_shape(pool1_3)
    print(shape)
    rep1 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool1_3)

    adaconv_all = layers.Concatenate()([rep1, rep3, rep5])
    #output_conv = layers.Dense(1, use_bias=False)(adaconv_all)
    #output_conv = layers.LeakyReLU(alpha=0.3)(output_conv)
    adaconv_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_all, state, subred_pred_embedded])
    adaconv_2 = adaptive_conv(time_distributed = True, filters=1, kernel_size=1, rank=1, padding='causal', activation='linear')([adaconv_all, state, subred_pred_embedded])
    output_conv = layers.LeakyReLU(alpha=0.3)(adaconv_2)
    output_conv = layers.Reshape((shape[1], 1))(output_conv)
    #adaconv_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_2, state, subred_pred_embedded])
    #output_conv = adaptive_conv(time_distributed = True, filters=1, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_3, state, subred_pred_embedded])
    #output = layers.Dense(1, activation='relu')(adaconv_all)
    rate_factor = layers.Dense(1, activation='sigmoid')(comment_rate)
    output = layers.Multiply()([rate_factor, output_conv])
    out = layers.Lambda(lambda x:K.exp(x))(output)
    return Model([news_input, sub_input, sub_predict, subred_input, subred_pred, comment_rate], out)

def build_model_observe(news_per_hour,
                token_per_news,
                sub_per_hour,
                token_per_sub):

    news_input = layers.Input(batch_shape=(1,
                                           news_per_hour,
                                           token_per_news))
    sub_input = layers.Input(batch_shape=(1,
                                          sub_per_hour,
                                          token_per_sub))
    sub_predict = layers.Input(batch_shape=(1,
                                            sub_per_hour,
                                            token_per_sub))
                                            
    comment_rate = layers.Input(batch_shape=(1,
                                             sub_per_hour,
                                             1))
                                            
    subred_input = layers.Input(batch_shape=(1,
                                            sub_per_hour,))                                        
    subred_pred = layers.Input(batch_shape=(1,
                                           sub_per_hour,))
                                                                                                                            
    embedding_layer = layers.Embedding(input_dim=emb_weights.shape[0],
                                       output_dim=100,
                                       weights = [emb_weights],
                                       trainable=True
                                       )
    subreddit_embedding = layers.Embedding(input_dim=44, output_dim=32)
    
    
    news_text = embedding_layer(news_input)
    sub_text = embedding_layer(sub_input)
    predict_text = embedding_layer(sub_predict)
    
    subred_input_embedded = subreddit_embedding(subred_input)
    subred_pred_embedded = subreddit_embedding(subred_pred)
    
    
    #sub_pred = layers.Concatenate(axis=-1)([sub_pred, subred_pred_embedded])
    

    N = layers.TimeDistributed(text_cnn(token_per_news,
                                        text_embedding_dim))(news_text)
    S = layers.TimeDistributed(text_cnn(token_per_sub,
                                        text_embedding_dim))(sub_text)
    S = layers.Concatenate(axis=-1)([S, subred_input_embedded])
    
    news_state = layers.GRU(128,
                                 #return_state=True,
                                 stateful=True)(N)
    sub_state = layers.GRU(128,
                                #return_state=True,
                                stateful=True)(S)
    state = layers.Concatenate(axis=-1)([news_state, sub_state])

    adaconv5_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=5, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool5_1 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_1)
    adaconv5_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=5, rank=1, padding='same', activation='relu')([pool5_1, state, subred_pred_embedded])
    pool5_2 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_2)
    adaconv5_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=5, rank=1, padding='same', activation='relu')([pool5_2, state, subred_pred_embedded])
    pool5_3 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_3)
    #flat5 = layers.Flatten()(pool5_3)
    shape = K.int_shape(pool5_3)
    rep5 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool5_3)
    
    adaconv3_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=3, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool3_1 = layers.TimeDistributed(layers.MaxPool1D(3, padding='same'))(adaconv3_1)
    adaconv3_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=3, rank=1, padding='same', activation='relu')([pool3_1, state, subred_pred_embedded])
    pool3_2 = layers.TimeDistributed(layers.MaxPool1D(3, padding='same'))(adaconv3_2)
    adaconv3_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=4, rank=1, padding='same', activation='relu')([pool3_2, state, subred_pred_embedded])
    pool3_3 = layers.TimeDistributed(layers.MaxPool1D(13, padding='same'))(adaconv3_3)
    #flat3 = layers.Flatten()(pool3_3)
    shape = K.int_shape(pool3_3)
    rep3 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool3_3)

    adaconv1_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=1, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool1_1 = layers.TimeDistributed(layers.MaxPool1D(1, padding='same'))(adaconv1_1)
    adaconv1_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=1, rank=1, padding='same', activation='relu')([pool1_1, state, subred_pred_embedded])
    pool1_2 = layers.TimeDistributed(layers.MaxPool1D(1, padding='same'))(adaconv1_2)
    adaconv1_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=1, rank=1, padding='same', activation='relu')([pool1_2, state, subred_pred_embedded])
    pool1_3 = layers.TimeDistributed(layers.MaxPool1D(150, padding='same'))(adaconv1_3)
    #flat1 = layers.Flatten()(pool1_3)
    shape = K.int_shape(pool1_3)
    print(shape)
    rep1 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool1_3)

    adaconv_all = layers.Concatenate()([rep1, rep3, rep5])
    #output_conv = layers.Dense(1, use_bias=False)(adaconv_all)
    #output_conv = layers.LeakyReLU(alpha=0.3)(output_conv)
    adaconv_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_all, state, subred_pred_embedded])
    adaconv_2 = adaptive_conv(time_distributed = True, filters=1, kernel_size=1, rank=1, padding='causal', activation='linear')([adaconv_all, state, subred_pred_embedded])
    #output_conv = layers.Reshape((shape[1], 1))(adaconv_2)
    
    ccount = layers.Input(batch_shape=(1, 50, 10, 1))
    observe = layers.Concatenate(axis=-2)([adaconv_2, ccount])
    output = layers.TimeDistributed(layers.SimpleRNN(1, activation='linear'))(observe)
    output = layers.LeakyReLU(alpha=0.3)(output)
    
    #adaconv_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_2, state, subred_pred_embedded])
    #output_conv = adaptive_conv(time_distributed = True, filters=1, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_3, state, subred_pred_embedded])
    #output = layers.Dense(1, activation='relu')(adaconv_all)
    rate_factor = layers.Dense(1, activation='sigmoid')(comment_rate)
    output = layers.Multiply()([rate_factor, output])
    out = layers.Lambda(lambda x:K.exp(x))(output)
    return Model([news_input, sub_input, sub_predict, subred_input, subred_pred, comment_rate, ccount], out)
    

def build_model_observeLSTM(news_per_hour,
                            token_per_news,
                            sub_per_hour,
                            token_per_sub,
                            comment_steps):

    news_input = layers.Input(batch_shape=(1,
                                           news_per_hour,
                                           token_per_news))
    sub_input = layers.Input(batch_shape=(1,
                                          sub_per_hour,
                                          token_per_sub))
    sub_predict = layers.Input(batch_shape=(1,
                                            sub_per_hour,
                                            token_per_sub))
                                            
    comment_rate = layers.Input(batch_shape=(1,
                                             sub_per_hour,
                                             1))
                                            
    subred_input = layers.Input(batch_shape=(1,
                                            sub_per_hour,))                                        
    subred_pred = layers.Input(batch_shape=(1,
                                           sub_per_hour,))
                                                                                                                            
    embedding_layer = layers.Embedding(input_dim=emb_weights.shape[0],
                                       output_dim=100,
                                       weights = [emb_weights],
                                       trainable=True
                                       )
    subreddit_embedding = layers.Embedding(input_dim=44, output_dim=32)
    
    
    news_text = embedding_layer(news_input)
    sub_text = embedding_layer(sub_input)
    predict_text = embedding_layer(sub_predict)
    
    subred_input_embedded = subreddit_embedding(subred_input)
    subred_pred_embedded = subreddit_embedding(subred_pred)
    
    
    #sub_pred = layers.Concatenate(axis=-1)([sub_pred, subred_pred_embedded])
    

    N = layers.TimeDistributed(text_cnn(token_per_news,
                                        text_embedding_dim))(news_text)
    S = layers.TimeDistributed(text_cnn(token_per_sub,
                                        text_embedding_dim))(sub_text)
    S = layers.Concatenate(axis=-1)([S, subred_input_embedded])
    
    news_state = layers.GRU(128,
                                 #return_state=True,
                                 stateful=True)(N)
    sub_state = layers.GRU(128,
                                #return_state=True,
                                stateful=True)(S)
    state = layers.Concatenate(axis=-1)([news_state, sub_state])

    adaconv5_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=5, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool5_1 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_1)
    adaconv5_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=5, rank=1, padding='same', activation='relu')([pool5_1, state, subred_pred_embedded])
    pool5_2 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_2)
    adaconv5_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=5, rank=1, padding='same', activation='relu')([pool5_2, state, subred_pred_embedded])
    pool5_3 = layers.TimeDistributed(layers.MaxPool1D(5, padding='same'))(adaconv5_3)
    #flat5 = layers.Flatten()(pool5_3)
    shape = K.int_shape(pool5_3)
    rep5 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool5_3)
    
    adaconv3_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=3, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool3_1 = layers.TimeDistributed(layers.MaxPool1D(3, padding='same'))(adaconv3_1)
    adaconv3_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=3, rank=1, padding='same', activation='relu')([pool3_1, state, subred_pred_embedded])
    pool3_2 = layers.TimeDistributed(layers.MaxPool1D(3, padding='same'))(adaconv3_2)
    adaconv3_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=4, rank=1, padding='same', activation='relu')([pool3_2, state, subred_pred_embedded])
    pool3_3 = layers.TimeDistributed(layers.MaxPool1D(13, padding='same'))(adaconv3_3)
    #flat3 = layers.Flatten()(pool3_3)
    shape = K.int_shape(pool3_3)
    rep3 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool3_3)

    adaconv1_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=1, rank=1, padding='same', activation='relu')([predict_text, state, subred_pred_embedded])
    pool1_1 = layers.TimeDistributed(layers.MaxPool1D(1, padding='same'))(adaconv1_1)
    adaconv1_2 = adaptive_conv(time_distributed = True, filters=64, kernel_size=1, rank=1, padding='same', activation='relu')([pool1_1, state, subred_pred_embedded])
    pool1_2 = layers.TimeDistributed(layers.MaxPool1D(1, padding='same'))(adaconv1_2)
    adaconv1_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=1, rank=1, padding='same', activation='relu')([pool1_2, state, subred_pred_embedded])
    pool1_3 = layers.TimeDistributed(layers.MaxPool1D(150, padding='same'))(adaconv1_3)
    #flat1 = layers.Flatten()(pool1_3)
    shape = K.int_shape(pool1_3)
    print(shape)
    rep1 = layers.Reshape((shape[1], 1, shape[2]*shape[3]))(pool1_3)

    adaconv_all = layers.Concatenate()([rep1, rep3, rep5])
    #output_conv = layers.Dense(1, use_bias=False)(adaconv_all)
    #output_conv = layers.LeakyReLU(alpha=0.3)(output_conv)
    adaconv_1 = adaptive_conv(time_distributed = True, filters=128, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_all, state, subred_pred_embedded])
    adaconv_2 = adaptive_conv(time_distributed = True, filters=1, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_all, state, subred_pred_embedded])
    #output_conv = layers.Reshape((shape[1], 1))(adaconv_2)
    rate_factor = layers.Dense(1, activation='sigmoid')(comment_rate)
    output = layers.Multiply()([rate_factor, adaconv_2])
    ccount = layers.Input(batch_shape=(1, sub_per_hour, comment_steps, 1))
    observe = layers.Concatenate(axis=-2)([output, ccount])
    interim = layers.TimeDistributed(layers.GRU(16))(observe)
    output = layers.Dense(1, activation='relu')(interim)
    #out = layers.LeakyReLU(alpha=0.2)(output)
    
    #adaconv_3 = adaptive_conv(time_distributed = True, filters=32, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_2, state, subred_pred_embedded])
    #output_conv = adaptive_conv(time_distributed = True, filters=1, kernel_size=1, rank=1, padding='causal', activation='relu')([adaconv_3, state, subred_pred_embedded])
    #output = layers.Dense(1, activation='relu')(adaconv_all)
    
    
    #out = layers.Lambda(lambda x:K.exp(x))(output)
    return Model([news_input, sub_input, sub_predict, subred_input, subred_pred, comment_rate, ccount], output)
