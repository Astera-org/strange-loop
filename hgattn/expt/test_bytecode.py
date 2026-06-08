import inspect
import operator
import dis

def evaluate_expr(expr):
    """
    Performs a simple stack-based formula evaluation.
    Keeps track of state.  But, the *full* state of the actual Python
    execution involves the instruction pointer and a lot of very complicated
    state.  
    """
    ops = dict(add=operator.add, sub=operator.sub, mul=operator.mul, div=operator.floordiv) 
    st = []
    for tok in expr:
        match tok:
            case int():
                st.append(tok)
            case str():
                op = ops[tok]
                arg2 = st.pop()
                arg1 = st.pop()
                result = op(arg1, arg2)
                st.append(result)
    ans = st.pop()
    return ans


expr = [4, 19, 'add', 25, 3, 'sub', 'mul']

print(evaluate_expr(expr))


def arith(a, b):
    return 3 * a - (4 + b)



def fib(n):
    a, b = 1, 1
    for _ in range(n):
        t = a + b
        a = b
        b = t
        # a, b = b, a + b
    return b

