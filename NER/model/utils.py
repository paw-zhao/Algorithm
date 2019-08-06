#-*- coding:utf-8 -*-
"""
@file: utils.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-7-15
"""
import pickle

def save_model(model, file_name):
    """用于保存模型"""
    with open(file_name, "wb") as f:
        pickle.dump(model, f)

def load_model(file_name):
    """用于加载模型"""
    with open(file_name, "rb") as f:
        model = pickle.load(f)
    return model

def flatten_lists(lists):
    flatten_list = []
    for l in lists:
        if type(l) is list:
            flatten_list += l
        else:
            flatten_list.append(l)
    return flatten_list

