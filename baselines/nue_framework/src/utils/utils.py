def get_problem_type(problem):
    """
    Function:
        get_problem_type
    Description:
        Safely returns a string as one of 4 problem times: regression, classification, both or none
    Input:
        - problem,str: Type of problem.
    Output:
        String representing type of problem, and none if not in the 4 types
    """
    if problem.lower() in ["regression", "classification", "both", "none"]:
        return problem.lower()
    else:
        return "none"

def argsort(list, *, key = None, reverse = False):
    new_list = [[i, e] for i, e in enumerate(list)]

    if key is None:
        fun = lambda r : r[1]
    else:
        fun = lambda r : key(r[1])
    
    new_list.sort(key = fun, reverse=reverse)
    arg = [ i for i, _ in new_list ]
    return arg


def sortarg(list, arg):
    return [ list[i] for i in arg ]

def normalize_score( score, metrics ):
    ret = []
    for s, m in zip(score, metrics):
        n = (s - m.lo) / (m.hi - m.lo)
        if m._sign == -1:
            n = 1 - n
        ret.append(n)
    return ret

if __name__ == "__main__":
    from random import randint

    x = [ randint(0, 20) for i in range(10) ]
    print(x)
    y = argsort(x)
    print(x)
    print(y)
    z = sortarg(x, y)
    print(z)

