#-*- coding:utf-8 -*-
"""
@file: NERData_Pre.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-8-1
"""
import os
import json
from codecs import open


BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR,'Input')
OUTPUT_DIR = os.path.join(BASE_DIR,'Output')
OUTPUTFILE = os.path.join(OUTPUT_DIR,'output_Pre.txt')


class DATAPre(object):
    """将JSON数据转换为标准的NER数据集"""
    def __init__(self,file):
        self.file = os.path.join(INPUT_DIR,file)
        self.data = []

    def dataTrans(self,sep='\t'):
        tag_lists = []
        with open(self.file,'r',encoding='utf-8') as load_f:
            load_dict = json.load(load_f)
            content = load_dict['content']
            annotation = load_dict['outputs']['annotation']['T'][1:]
            #对标注进行排序
            annotation = sorted(annotation,key=lambda x:x['start'])

            #对标注进行合并
            n = len(annotation)
            index = []
            for i in range(n-1):
                tag_pre = annotation[i]['name']
                tag_post = annotation[i+1]['name']
                end_pre = annotation[i]['end']
                start_post = annotation[i+1]['start']
                if tag_pre == tag_post and end_pre == start_post :
                    annotation[i+1]['start'] = annotation[i]['start']
                    index.append(i)
            for j in index:
                del annotation[j]

            word_lsts = [word for word in content if word != '\r']
            #处理tag_lists
            start_pre = 0
            noun = ['B','M','E','S','O']
            for item in annotation:
                if item is not None:
                    start = item['start']
                    end = item['end']
                    while start != start_pre:
                        tag_lists.append(noun[4])
                        start_pre += 1
                    length = end - start
                    if length == 1:
                        tag_lists.append(noun[3] + '-' + item['name'])
                    else:
                        tag_lists.append(noun[0] + '-' + item['name'])
                        for i in range(start+1,end-1):
                            tag_lists.append(noun[1] + '-' + item['name'])
                        tag_lists.append(noun[2] + '-' + item['name'])
                    start_pre = end

        with open(OUTPUTFILE, 'w', encoding='utf-8') as dump_f:
            for item1,item2 in zip(word_lsts,tag_lists):
                if item1 != '\n':
                    dump_f.write(item1 + sep + item2 + '\n')
                else:
                    dump_f.write('\n')


    def devProc(self):
        with open(self.file,'r',encoding='utf-8') as load_f:
            for line in load_f:
                for word in line:
                    if word != '\r':
                        self.data.append(word)

        with open(OUTPUTFILE, 'w', encoding='utf-8') as dump_f:
            for word in self.data:
                dump_f.write(word.strip('\n')+'\n')


if __name__ == '__main__':
    instance = DATAPre('dev.txt')
    instance.devProc()



