#-*- coding:utf-8 -*-
"""
@file: main.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-7-15
"""
from data import build_corpus
from evaluate import hmm_train_eval
from utils import load_model,flatten_lists
import pandas as pd
from codecs import open

HMM_MODEL_PATH = './CKPT/hmm.pkl'

REMOVE_O = False  # 在评估的时候是否去除O标记

def train_eval():
    """训练模型，评估结果"""
    # 读取数据
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train.txt",sep=' ')
    test_word_lists, test_tag_lists = build_corpus("test.txt", make_vocab=False,sep='\t')

    # 训练评估ｈｍｍ模型
    print("正在训练评估HMM模型...")
    hmm_train_eval(
        (train_word_lists, train_tag_lists),
        (test_word_lists, test_tag_lists),
        word2id,
        tag2id
    )

def predict(sep='\t'):
    """预测结果"""
    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = \
        build_corpus("train.txt",sep=' ')
    dev_word_lists = build_corpus("dev.txt")

    print("加载模型...")
    hmm_model = load_model(HMM_MODEL_PATH)
    hmm_pred = hmm_model.test(dev_word_lists,
                              word2id,
                              tag2id)
    #保存数据
    result = pd.DataFrame()
    result['word'] = flatten_lists(dev_word_lists)
    result['pred_tag'] = flatten_lists(hmm_pred)
    with open('./Output/result.txt','w',encoding='utf-8') as f:
        for word,tag in zip(result['word'],result['pred_tag']):
            f.write(word+sep+tag+'\n')
    print('完成！')

if __name__ == '__main__':
    # train_eval()
    predict(sep='\t')
