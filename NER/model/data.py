#-*- coding:utf-8 -*-
"""
@file: data.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-7-15
"""
from os.path import join
from codecs import open


def build_corpus(split, make_vocab=True, data_dir="./Input",sep='\t'):
    """读取数据"""
    assert split in ['train.txt','test.txt','dev.txt']
    word_lists = []
    tag_lists = []
    if split == 'dev.txt':
        with open(join(data_dir, split), 'r', encoding='utf-8') as f:
            word_list = []
            for line in f:
                if line != '\n':
                    word = line.strip('\n')
                    word_list.append(word)
                else:
                    word_lists.append(word_list)
                    word_list = []
        return word_lists

    else:
        with open(join(data_dir, split), 'r', encoding='utf-8') as f:
            word_list = []
            tag_list = []
            for line in f:
                if line != '\n':
                    word, tag = line.strip('\n').split(sep)
                    word_list.append(word)
                    tag_list.append(tag)
                else:
                    word_lists.append(word_list)
                    tag_lists.append(tag_list)
                    word_list = []
                    tag_list = []

        # 如果make_vocab为True，还需要返回word2id和tag2id
        if make_vocab:
            word2id = build_map(word_lists)
            tag2id = build_map(tag_lists)
            return word_lists, tag_lists, word2id, tag2id
        else:
            return word_lists, tag_lists

def build_map(lists):
    maps = {}
    for list_ in lists:
        for e in list_:
            if e not in maps:
                maps[e] = len(maps)

    return maps
