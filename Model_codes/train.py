import json
import numpy as np
from build_model import build_model_observeLSTM, masked_relative_error
from keras.optimizers import Adam
n_input = np.load('../Data_processing/news_text.npy')[:-1]
submission = np.load('../Data_processing/submission_text_october.npy')[:len(n_input)+1]
comment_count = np.load('../Data_processing/temporal_cc10min_october.npy')[:len(n_input)+1]
subred = np.load('../Data_processing/submission_subred_october.npy')[:len(n_input)+1]
comment_rate = np.load('../Data_processing/submission_comment_rate_october.npy')[:len(n_input)]
comment_rate = np.reshape(comment_rate, comment_rate.shape+(1,))
s_input = submission[:-1]
c_count = comment_count[1:]
c_count = c_count.reshape((c_count.shape[0], c_count.shape[1], c_count.shape[2], 1))
s_pred = submission[1:]
subred_input = subred[:-1]
subred_pred = subred[1:]
s_value = np.load('../Data_processing/submission_value_new_october.npy')[1:len(n_input)+1]
_, news_per_hour, token_per_news = n_input.shape
_, sub_per_hour, token_per_sub = s_input.shape
comment_steps = c_count.shape[-2]
model = build_model_observeLSTM(news_per_hour,
                    token_per_news,
                    sub_per_hour,
                    token_per_sub,
                                comment_steps)
model.compile(loss=masked_relative_error(-1.), optimizer=Adam(lr=0.00001, clipnorm=1.0))
for i in range(20):
    model.fit([n_input,
               s_input,
               s_pred,
               subred_input,
               subred_pred,
               comment_rate,
               c_count],
                    s_value,
                    batch_size=1,
                    epochs=1)
    if i<19:
        model.reset_states()
    model.save('model_observeLSTM_exp-gru.h5')

