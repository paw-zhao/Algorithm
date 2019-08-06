#-*- coding:utf-8 -*-
"""
@file: list_recur.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-7-4
"""
class TreeNode:
    def __init__(self, data=-1, lchild=None, rchild=None):
        self.lchild = lchild  # 表示左子树
        self.rchild = rchild  # 表示右子树
        self.data = data  # 表示数据域

class Tree:
    # 递归建树
    def traversal_create(self,root,lst,i):
        if i < len(lst):
            if lst[i] is '#':
                return
            else:
                root = TreeNode(data=lst[i])
                root.lchild = self.traversal_create(root.lchild,lst,2*i + 1)
                root.rchild = self.traversal_create(root.rchild,lst,2*i + 2)
                return root
        return root

    #前序递归
    def preorder(self,root):
        if root is None:
            return
        print(root.data)
        self.preorder(root.lchild)
        self.preorder(root.rchild)

if __name__ == "__main__":
    root = TreeNode()
    lst = [10,8,12,6,9,11,13,'#',7,'#','#','#','#','#',15]
    t = Tree()
    d = t.traversal_create(root,lst,0)
    t.preorder(d)


