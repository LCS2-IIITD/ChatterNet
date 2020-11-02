import json
import numpy as np
from build_model import build_model_observeLSTM, masked_relative_error
from matplotlib import pyplot as plt
from scipy.stats import kendalltau, spearmanr
def smape(A, F):
    return (100/len(A)) * np.sum((2 * np.abs(F - A)) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))
def mape(A, F):
    return (100/len(A)) * np.sum(np.abs(F - A) / (np.abs(A) + np.finfo(float).eps))
n_input = np.load('../Data_processing/news_text_november.npy')[:-1]
print(n_input.shape)
submission = np.load('../Data_processing/submission_text_november.npy')[:len(n_input)+1]
comment_count = np.load('../Data_processing/temporal_cc60min_november.npy')[:len(n_input)+1]
subred = np.load('../Data_processing/submission_subred_november.npy')[:len(n_input)+1]
comment_rate = np.load('../Data_processing/submission_comment_rate_november.npy')[:len(n_input)]
comment_rate = np.reshape(comment_rate, comment_rate.shape+(1,))
s_input = submission[:-1]
c_count = comment_count[1:]
c_count = c_count.reshape((c_count.shape[0], c_count.shape[1], c_count.shape[2], 1))
s_pred = submission[1:]
subred_input = subred[:-1]
subred_pred = subred[1:]
s_value = np.load('../Data_processing/submission_value_60min_november.npy')[1:len(n_input)+1]
_, news_per_hour, token_per_news = n_input.shape
_, sub_per_hour, token_per_sub = s_input.shape
comment_steps = c_count.shape[-2]

model = build_model_observeLSTM(news_per_hour,
                    token_per_news,
                    sub_per_hour,
                    token_per_sub,
                                comment_steps)
model.load_weights('model_observeLSTM_exp-gru.h5')
model.compile(loss=masked_relative_error(0.), optimizer='adam')
pred_value = model.predict([n_input, s_input, s_pred, subred_input, subred_pred, comment_rate, c_count], batch_size=1, verbose=1)

y_true = np.reshape(s_value, (s_value.shape[0]*s_value.shape[1]*s_value.shape[2]))
y_pred = np.reshape(pred_value, (s_value.shape[0]*s_value.shape[1]*s_value.shape[2]))

A, F = [], []
for i in range(len(y_true)):
    if y_true[i]!=-1.:
        A.append(y_true[i])
        F.append(y_pred[i])

print('sMAP Error:', smape(A, F))
print('MAP Error:', mape(A,F))
tau, _ = kendalltau(A,F)
rho, _ = spearmanr(A,F)
print('kendal tau:', tau)
print('spearman rho:', rho)
