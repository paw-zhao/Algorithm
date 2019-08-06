#-*- coding:utf-8 -*-
"""
@file: level_create.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-7-2
"""
class TreeNode:
    def __init__(self,val):
        self.item = val
        self.lchild = None
        self.rchild = None

class Tree:
    def __init__(self):
        self.root = None

    def add(self,val):
        node = TreeNode(val)
        if self.root is None:
            self.root = node
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            if cur_node.lchild is None:
                cur_node.lchild = node
                return

            elif cur_node.rchild is None:
                cur_node.rchild = node
                return

            else:
                queue.append(cur_node.lchild)
                queue.append(cur_node.rchild)

    def preorder(self,node):
        if node is None:
            return
        print(node.item)
        self.preorder(node.lchild)
        self.preorder(node.rchild)

    def inorder(self,node):
        if node is None:
            return
        self.inorder(node.lchild)
        print(node.item)
        self.inorder(node.rchild)

    def postorder(self,node):
        if node is None:
            return
        self.postorder(node.lchild)
        self.postorder(node.rchild)
        print(node.item)

    def beadth_order(self):
        if self.root is None:
            return
        queue = [self.root]
        while queue:
            cur_node = queue.pop(0)
            print(cur_node.item)
            if cur_node.lchild is not None:
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)

t = Tree()
for i in range(10):
    t.add(i)
t.beadth_order()






