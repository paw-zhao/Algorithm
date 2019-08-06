#-*- coding:utf-8 -*-
"""
@file: HMM.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-5-6
"""
import torch

class HMM:
   def __init__(self,N,M):
       """
       :param N: 状态数
       :param M: 观测数
       """
       self.N = N
       self.M = M

       #初始状态概率
       self.Pi = torch.zeros(N)
       #状态转移概率矩阵
       self.A = torch.zeros([N,N])
       #观测概率矩阵
       self.B = torch.zeros([N,M])

   def train(self, word_lists, tag_lists, word2id, tag2id):
       """
       :param word_lists: 列表，其中每个元素由字组成的列表，如 ['担','任','科','员']
       :param tag_lists:列表，其中每个元素是由对应的标注组成的列表，如 ['O','O','B-TITLE', 'E-TITLE']
       :param word2id:将字映射为ID
       :param tag2id:将标注映射为ID
       """
       assert len(tag_lists) == len(word_lists)
       # 估计初始状态概率
       for tag_list in tag_lists:
           init_tagid = tag2id[tag_list[0]]
           self.Pi[init_tagid] += 1
       self.Pi[self.Pi == 0.] = 1e-10
       self.Pi = self.Pi / self.Pi.sum()

       #估计转移概率矩阵
       for tag_list in tag_lists:
           seq_len = len(tag_list)
           for i in range(seq_len-1):
               current_tagid = tag2id[tag_list[i]]
               next_tagid = tag2id[tag_list[i+1]]
               self.A[current_tagid][next_tagid] += 1
       self.A[self.A == 0.] = 1e-10
       self.A = self.A / self.A.sum(dim=1,keepdim=True)

       #估计观测概率矩阵
       for tag_list,word_list in zip(tag_lists,word_lists):
           assert len(tag_list) == len(word_list)
           for tag,word in zip(tag_list,word_list):
               tag_id = tag2id[tag]
               word_id = word2id[word]
               self.B[tag_id][word_id] += 1
       self.B[self.B == 0.] = 1e-10
       self.B = self.B / self.B.sum(dim=1,keepdim=True)

   def test(self,word_lists,word2id,tag2id):
       pred_tag_lists = []
       for word_list in word_lists:
           pred_tag_list = self.decoding(word_list,word2id,tag2id)
           pred_tag_lists.append(pred_tag_list)
       return pred_tag_lists

   def decoding(self, word_list, word2id, tag2id):
       """
               使用维特比算法对给定观测序列求状态序列， 这里就是对字组成的序列,求其对应的标注。
               维特比算法实际是用动态规划解隐马尔可夫模型预测问题，即用动态规划求概率最大路径（最优路径）
               这时一条路径对应着一个状态序列
               """
       # 问题:整条链很长的情况下，十分多的小概率相乘，最后可能造成下溢
       # 解决办法：采用对数概率，这样源空间中的很小概率，就被映射到对数空间的大的负数
       #  同时相乘操作也变成简单的相加操作
       A = torch.log(self.A)
       B = torch.log(self.B)
       Pi = torch.log(self.Pi)

       # 初始化
       seq_len = len(word_list)
       viterbi = torch.zeros(self.N, seq_len)
       backpointer = torch.zeros(self.N, seq_len).long()

       start_wordid = word2id.get(word_list[0], None)
       Bt = B.t()
       if start_wordid is None:
           bt = torch.log(torch.ones(self.N) / self.N)
       else:
           bt = Bt[start_wordid]
       viterbi[:, 0] = Pi + bt
       backpointer[:, 0] = -1

       # 递推
       for step in range(1, seq_len):
           wordid = word2id.get(word_list[step], None)
           # 处理字不在字典中的情况
           if wordid is None:
               bt = torch.log(torch.ones(self.N) / self.N)
           else:
               bt = Bt[wordid]
           for tag_id in range(len(tag2id)):
               max_prob, max_id = torch.max(
                   viterbi[:, step - 1] + A[:, tag_id],
                   dim=0
               )
               viterbi[tag_id, step] = max_prob + bt[tag_id]
               backpointer[tag_id, step] = max_id

       # 终止
       best_path_prob, best_path_pointer = torch.max(
           viterbi[:, seq_len - 1],
           dim=0
       )

       # 回溯，求最优路径
       best_path_pointer = best_path_pointer.item()
       best_path = [best_path_pointer]
       for back_step in range(seq_len - 1, 0, -1):
           best_path_pointer = backpointer[best_path_pointer, back_step]
           best_path_pointer = best_path_pointer.item()
           best_path.append(best_path_pointer)

       # 将tag_id组成的序列转化为tag
       assert len(best_path) == len(word_list)
       id2tag = dict((id_, tag) for tag, id_ in tag2id.items())
       tag_list = [id2tag[id_] for id_ in reversed(best_path)]

       return tag_list




       









