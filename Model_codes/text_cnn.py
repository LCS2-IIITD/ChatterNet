from keras import layers
from keras.models import Model

def text_cnn(token_per_text, embedding_dim):
    i_emb = layers.Input(shape=(token_per_text, embedding_dim))
    conv5_1 = layers.Conv1D(128, 5, padding='same', activation='relu')(i_emb)
    pool5_1 = layers.MaxPool1D(5, padding='same')(conv5_1)
    conv5_2 = layers.Conv1D(64, 5, padding='same', activation='relu')(pool5_1)
    pool5_2 = layers.MaxPool1D(5, padding='same')(conv5_2)
    conv5_3 = layers.Conv1D(32, 5, padding='same', activation='relu')(pool5_2)
    pool5_3 = layers.MaxPool1D(5, padding='same')(conv5_3)
    flat5 = layers.Flatten()(pool5_3)
    
    conv3_1 = layers.Conv1D(128, 3, padding='same', activation='relu')(i_emb)
    pool3_1 = layers.MaxPool1D(3, padding='same')(conv3_1)
    conv3_2 = layers.Conv1D(64, 3, padding='same', activation='relu')(pool3_1)
    pool3_2 = layers.MaxPool1D(3, padding='same')(conv3_2)
    conv3_3 = layers.Conv1D(32, 4, padding='same', activation='relu')(pool3_2)
    pool3_3 = layers.MaxPool1D(13, padding='same')(conv3_3)
    flat3 = layers.Flatten()(pool3_3)

    conv1_1 = layers.Conv1D(128, 1, padding='same', activation='relu')(i_emb)
    pool1_1 = layers.MaxPool1D(1, padding='same')(conv1_1)
    conv1_2 = layers.Conv1D(64, 1, padding='same', activation='relu')(pool1_1)
    pool1_2 = layers.MaxPool1D(1, padding='same')(conv1_2)
    conv1_3 = layers.Conv1D(32, 1, padding='same', activation='relu')(pool1_2)
    pool1_3 = layers.MaxPool1D(150, padding='same')(conv1_3)
    flat1 = layers.Flatten()(pool1_3)
    
    o = layers.Concatenate()([flat5, flat3, flat1])

    return Model(i_emb,o)
