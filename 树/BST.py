#-*- coding:utf-8 -*-
"""
@file: BST.py
@version:v1.0
@software:PyCharm

@author: fenglong.zhao
@contact: fenglong.zhao@irootech.com
@time:2019-7-4
"""
class Node:
    def __init__(self,val=-1):
        self.data = val
        self.lchild = None
        self.rchild = None

class Tree:
    def __init__(self):
        self.count = 0

    def insert(self,root,val):
        if root is None:
            root = Node(val)
        elif val <= root.data:
            root.lchild = self.insert(root.lchild,val)
        elif val > root.data:
            root.rchild = self.insert(root.rchild,val)
        return root

    def breadth_order(self,root):
        if root is None:
            return None
        queue = [root]
        while queue:
            cur_node = queue.pop(0)
            print(cur_node.data)
            if cur_node.lchild is not None:
                queue.append(cur_node.lchild)
            if cur_node.rchild is not None:
                queue.append(cur_node.rchild)

    def preorder(self,root):
        if root is None:
            return None
        print(root.data)
        self.preorder(root.lchild)
        self.preorder(root.rchild)

    def inorder(self,root):
        if root is None:
            return None
        self.inorder(root.lchild)
        print(root.data)
        self.inorder(root.rchild)

    def postorder(self,root):
        if root is None:
            return None
        self.postorder(root.lchild)
        self.postorder(root.rchild)
        print(root.data)

    def query(self,root,val):
        if root is not None:
            if val == root.data:
                return True
            elif val < root.data:
                return self.query(root.lchild,val)
            else:
                return self.query(root.rchild,val)
        else:
            return False

    def minimum(self,root):
        if root.lchild is None:
            return root.data
        return self.minimum(root.lchild)

    def maximum(self,root):
        if root.rchild is None:
            return root.data
        return self.maximum(root.rchild)

    def delNode(self,root,val):
        if root is None:
            return
        if val < root.data:
            root.lchild = self.delNode(root.lchild,val)
        elif val > root.data:
            root.rchild = self.delNode(root.rchild,val)
        else:
            if root.lchild and root.rchild:
                tem = self.minimum(root.rchild)
                root.data = tem
                root.rchild = self.delNode(root.rchild,tem)
            elif not root.lchild and not root.rchild:
                root = None
            elif root.lchild:
                root = root.lchild
            elif root.rchild:
                root = root.rchild

        return root

    def modify(self,root,val,val_rep):
        if root is None:
            return
        if val < root.data:
            root.lchild = self.modify(root.lchild,val,val_rep)
        elif val > root.data:
            root.rchild = self.modify(root.rchild,val,val_rep)
        else:
            root.data = val_rep

        return root

    def node_number(self,root):
        if root is None:
            return 0
        self.count += 1
        self.node_number(root.lchild)
        self.node_number(root.rchild)
        return self.count

    def depth(self,root):
        if root is None:
            return 0
        ldepth = self.depth(root.lchild)
        rdepth = self.depth(root.rchild)
        return max(ldepth,rdepth) + 1

    def width(self,root):
        if root is None:
            return 0
        curWidth = 1
        maxWidth = 0
        queue = [root]
        while queue:
            n = curWidth
            curWidth = 0
            for i in range(n):
                tem = queue.pop(0)
                if tem.lchild is not None:
                    curWidth += 1
                    queue.append(tem.lchild)
                if tem.rchild is not None:
                    curWidth += 1
                    queue.append(tem.rchild)
            if curWidth > maxWidth:
                maxWidth = curWidth

        return maxWidth

    #递归法判断BBT树
    def is_BBT_recur(self,root):
        if root is None:
            return True
        leftheight = self.depth(root.lchild)
        rightheight = self.depth(root.rchild)
        if abs(leftheight-rightheight) > 1:
            return False
        return self.is_BBT_recur(root.lchild) and self.is_BBT_recur(root.rchild)

    #后序遍历法判断BBT树
    def is_BBT_post(self,root):
        return self.getDepth(root) != -1

    def getDepth(self,root):
        if root is None:
            return 0
        leftheight = self.getDepth(root.lchild)
        if leftheight == -1:
            return -1
        rightheight = self.getDepth(root.rchild)
        if rightheight == -1:
            return -1
        if abs(leftheight-rightheight) > 1:
            return -1
        else:
            return max(leftheight,rightheight) + 1

if __name__ == '__main__':
    #创建BST
    root = Node(17)
    t = Tree()
    lst = [5,35,2,16,29,38,19,33]
    for i in lst:
        root = t.insert(root,i)

    #插入节点
    root = t.insert(root,15)

    #查询
    # val = 33
    # print(t.query(root,val))

    #查找BST中最大值
    # print(t.maximum(root))

    #查找BST中最小值
    # print(t.minimum(root))

    #遍历
    # t.preorder(root)
    # t.breadth_order(root)
    # t.inorder(root)
    # t.postorder(root)

    #删除节点
    # val = 5
    # node = t.delNode(root,val)
    # t.breadth_order(node)

    #修改节点值
    # val = 38
    # val_rep = 40
    # node = t.modify(root,val,val_rep)
    # t.breadth_order(node)

    #求节点的个数
    # print(t.node_number(root))

    #求深度
    # print(t.depth(root))

    #求宽度
    # print(t.width(root))

    #怎么判断是否为BBT树
    # print(t.is_BBT_recur(root))

    #AVL树的实现
    











