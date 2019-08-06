#-*- coding:utf-8 -*-
"""
@file: NERData_Ext.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-8-2
"""
import os
from codecs import open
from collections import defaultdict


BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR,'Input')
OUTPUT_DIR = os.path.join(BASE_DIR,'Output')
OUTPUTFILE = os.path.join(OUTPUT_DIR,'output_Ext.txt')


class DATAExt(object):
    """对标注数据提取对应实体"""
    def __init__(self,file):
        self.file = os.path.join(INPUT_DIR,file)
        self.Entity = defaultdict(set)

    def dataExt(self,sep='\t'):
        noun = ['B', 'M', 'E', 'S', 'O']
        tag_name = ''
        with open(self.file,'r',encoding='utf-8') as f:
            for line in f:
                if line != '\r\n':
                    word,tag = line.strip('\n\r').split(sep)
                    if tag != noun[4]:
                        tag_name += word
                        if tag[:1] == noun[2] or tag[:1] == noun[3]:
                            self.Entity[tag[2:]].add(tag_name)
                            tag_name = ''

        return self.Entity

if __name__ == '__main__':
    instance = DATAExt('test3.txt')
    print(instance.dataExt(sep=' '))
