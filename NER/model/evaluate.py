#-*- coding:utf-8 -*-
"""
@file: evaluate.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-7-15
"""
from HMM import HMM
from utils import save_model
from evaluating import Metrics

def hmm_train_eval(train_data,test_data,word2id,tag2id,remove_O = False):
    """训练并评估模型"""
    #训练HMM模型
    train_word_lists,train_tag_lists = train_data
    test_word_lists,test_tag_lists  = test_data

    hmm_model = HMM(len(tag2id),len(word2id))
    hmm_model.train(train_word_lists,
                    train_tag_lists,
                    word2id,
                    tag2id)
    save_model(hmm_model,'./CKPT/hmm.pkl')

    #评估模型
    pred_tag_lists = hmm_model.test(test_word_lists,
                                    word2id,
                                    tag2id)

    metrics = Metrics(test_tag_lists,pred_tag_lists,remove_O=remove_O)
    metrics.report_scores()
    metrics.report_confusion_matrix()



