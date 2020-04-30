#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Various useful functions (work in progress)

@author: arno
@Date: Mar 2020
"""
import numpy as np


# =============================================================================
# Enhanced division by zero
# Used for the Random Forrest study to calculate trade intensity acceleration       
# ============================================================================= 
def divisionByZero(x,y):
    try:
        return x/y
    except ZeroDivisionError:
        return 1

        
# =============================================================================
# Extract decision rules from Random Forest
# Adapted from: https://stackoverflow.com/questions/50600290/how-extraction-decision-rules-of-random-forest-in-python       
# =============================================================================  
def getDecisionRules(rf):
    decisionRules = [] 
    for tree_idx, est in enumerate(rf.estimators_):
        tree = est.tree_
        assert tree.value.shape[1] == 1 # no support for multi-output

#        print('TREE: {}'.format(tree_idx))
#        decisionRules.append('TREE: {}'.format(tree_idx))

        iterator = enumerate(zip(tree.children_left, tree.children_right, tree.feature, tree.threshold, tree.value))
        for node_idx, data in iterator:
            left, right, feature, th, value = data

            # left: index of left child (if any)
            # right: index of right child (if any)
            # feature: index of the feature to check
            # th: the threshold to compare against
            # value: values associated with classes            

            # for classifier, value is 0 except the index of the class to return
            class_idx = np.argmax(value[0])

            if left == -1 and right == -1:
#                print('{} LEAF: return class={}'.format(node_idx, class_idx))
                decisionRules.append('{} LEAF: return class={}'.format(node_idx, class_idx))
            else:
#                print('{} NODE: if feature[{}] < {} then next={} else next={}'.format(node_idx, feature, th, left, right))    
                decisionRules.append('{} NODE: if feature[{}] < {} then next={} else next={}'.format(node_idx, feature, th, left, right))
    
    return(decisionRules)







    