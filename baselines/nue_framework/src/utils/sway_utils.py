import numpy as np
from math import exp


def distance_numeric_norm(df, col, i, j):
    c = df[col]
    low, high = min(c), max(c)
    return abs(float(c[i]) - float(c[j])) / ( high - low )

def distance_str(df, col, i, j):
    return 0 if df[col][i] == df[col][j] else 1

def distance_pair(df, types, i, j, p = 2):
    """
        Calculate distance between items i and j
        p is power, p = 2 is euclidean, etc
    """
    d = 0
    for t, v in types.items():
        f = distance_str
        if v in ["f", "i"]:
            f = distance_numeric_norm
        d += pow(f( df, t, i, j ), p)
    d /= len(types)
    d = pow(d, 1/p)
    return d

def distance_from(df, types, idx, i, p = 2):
    return [ distance_pair(df, types, i, j, p = p) for j in idx ]

def binary_dominates(a, b):
    """Returns whether a binary dominates b"""
    for ai, bi in zip(a, b):
        if bi > ai:
            return False
    if a == b:
        return False
    return True

def zitler_dominates(a, b):
    """Returns wether a zitler dominates b
    Requires a and b to be normalized in [0, 1] range
    1: a dominates b
    -1: a is dominated by b
    0: Neither dominates, equivalent"""
    s1, s2 = 0, 0
    n = len(a)
    for ai, bi in zip(a, b):
        s1 -= exp( (ai - bi)/n )
        s2 -= exp( (bi - ai)/n )
    if s1/n < s2/n:
        return 1
    elif s1/n > s2/n:
        return -1
    else:
        return 0
