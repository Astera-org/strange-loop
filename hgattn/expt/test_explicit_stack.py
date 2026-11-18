import fire
from dataclasses import dataclass
from typing import Callable
import numpy as np
from numpy.random import Generator
from enum import Enum, auto

class Node:
    def __init__(self, level: int, child_index: int):
        self.id = f"{level}_{child_index}"
        self.hit = False 
        self.children = []

    def __repr__(self):
        return f"{self.id}: hit: {self.hit}"


def sample_from_counts(rng, counts):
    csum = np.cumsum(counts)
    probe = rng.choice(csum[-1])
    return np.searchsorted(csum, probe, side="right")

def _build_tree(
    rng: Generator, 
    child_counts: np.array, 
    level: int, 
    child_index: int, 
    depth_to_go: int,
) -> Node:
    if child_counts.size == 0:
        n_child = 0
    else:
        n_child = sample_from_counts(rng, child_counts)

    if n_child == 0 or depth_to_go == 0:
        return Node(level, child_index)

    node = Node(level, child_index)
    for idx in range(n_child):
        child = _build_tree(rng, child_counts, level+1, idx, depth_to_go-1)
        node.children.append(child)

    return node

def _get_random_leaf(node: Node, rng) -> Node:
    if len(node.children) == 0:
        return node
    child = rng.choice(node.children)
    return _get_random_leaf(child, rng)

def build_tree(rng, child_counts, max_depth) -> Node:
    root = _build_tree(rng, child_counts, 0, 0, max_depth)
    leaf = _get_random_leaf(root, rng)
    leaf.hit = True
    return root



# Original function
def search(node: Node, rng: Generator) -> Node|None:
    if node.hit:      # enter 
        return node

    perm = rng.permutation(len(node.children)) 
    for p in perm:             # loop
        ch = node.children[p]
        hit = search(ch, rng)
        if hit is not None:    # after child
            return hit 
    return None

"""
Activation Record Reification technique.

This function will mimic the state transition that the hardware stack undergoes
between recursive calls of the original function.

One optimization however, is, instead of unwinding the stack upon a hit discovery,
a 'found' flag is set.

Also, in order to be jit-compilable, the final enter_node function must be a
single-step state machine.
"""

class Site(Enum):
    START = 0
    AFTER_CHILD = 1

@dataclass
class Frame:
    node: Node | None = None
    perm: np.ndarray | None = None
    loop_idx: int = 0
    found: bool = False
    in_backtrack: bool = False

def update_stack(stack: list[Frame], rng) -> list[Frame]:
    fr = stack[-1]

    if fr.in_backtrack:
        return fixed_backtrack(stack)

    if fr.node.hit:
        fr.found = True
        return stack

    # entering node for the first time
    if fr.perm is None:
        fr.perm = rng.permutation(len(fr.node.children))
        fr.loop_idx = 0

    if fr.loop_idx < fr.perm.size:
        return visit_child(stack)
    else:
        return fixed_backtrack(stack)


def fixed_backtrack(stack: list[Frame]) -> list[Frame]:
    """
    Backtrack at most n times, checking for a next unvisited child at each node.
    If one exists, update the parent's loop_idx to point to that child.
    Set the parent's `in_backtrack` flag to indicate whether more backtracking is
    pending.
    """
    found = False
    step = 0
    while stack and step < 2:
        fr = stack[-1]
        if fr.loop_idx < fr.perm.size - 1:
            fr.loop_idx += 1
            fr.in_backtrack = False
            return stack
        else:
            stack.pop()
            step += 1

    if stack:
        stack[-1].in_backtrack = True 
    return stack

def visit_child(stack: list[Frame]) -> list[Frame]:
    fr = stack[-1]
    p = fr.perm[fr.loop_idx]
    ch = fr.node.children[p]
    stack.append(Frame(ch))
    return stack

def next_node(stack: list[Frame]) -> list[Frame]:
    """Advance to the next in-order node, or empty the stack if there is none"""
    if not stack:
        return stack

    stack.pop()
    while stack:
        fr = stack[-1]
        if fr.loop_idx < fr.perm.size - 1: # not last child
            fr.loop_idx += 1
            return visit_child(stack)
        else:
            stack.pop()
    return stack


def search_loop(node, rng) -> Node | None:
    stack = [Frame(node)]
    iters = 0
    while stack and not stack[-1].found:
        stack = update_stack(stack, rng)
        iters += 1

    if len(stack):
        return stack[-1].node
    return None




"""
Conversion to continuation-passing style, using a pair of continuations,
success and failure
"""

def search_sub(node: Node, rng: Generator, ksucc, kfail) -> Node|None:
    if node.hit:
        return ksucc(node)
    perm = rng.permutation(len(node.children))
    return loop_step(node, perm, 0, rng, ksucc, kfail)

def loop_step(node, perm, i, rng, ksucc, kfail) -> Node|None:
    if i == len(perm):
        return kfail() 
    ch = node.children[perm[i]]
    kfail_next = lambda: loop_step(node, perm, i + 1, rng, ksucc, kfail)
    return search_sub(ch, rng, ksucc, kfail_next)

def search_cps(node, rng) -> Node|None:
    ksucc = lambda x: x
    kfail = lambda: None
    return search_sub(node, rng, ksucc, kfail)

"""
Defunctionalization of search_cps.
"""
class FailCont:
    pass

class FailRoot(FailCont):
    pass

@dataclass
class FailLoopStep(FailCont):
    node: Node
    perm: np.ndarray
    i: int
    parent: FailCont


def apply_fail(cont: FailCont, succ, rng: Generator) -> Node | None:
    if isinstance(cont, FailRoot):
        return None
    elif isinstance(cont, FailLoopStep):
        return loop_step_d(cont.node, cont.perm, cont.i, rng, succ, cont.parent)

def search_sub_d(node: Node, rng: Generator, ksucc, kfail) -> Node|None:
    if node.hit:
        return ksucc(node)
    perm = rng.permutation(len(node.children))
    return loop_step_d(node, perm, 0, rng, ksucc, kfail)

def loop_step_d(node, perm, i, rng, ksucc, kfail) -> Node|None:
    if i == len(perm):
        return apply_fail(kfail, rng)
    ch = node.children[perm[i]]
    kfail_next = FailLoopStep(node, perm, i+1, kfail)
    return search_sub_d(ch, rng, ksucc, kfail_next)

def search_cps_d(
    node: Node,
    rng: Generator,
) -> Node|None:
    ksucc = lambda x: x  # ksucc creation site 1
    return search_sub_d(node, rng, ksucc, FailRoot())


"""
Now, convert the reified stack from a linked-list structure to a stack structure  
"""
def search_sub_st(stack: list[Frame], rng) -> list[Frame]:
    fr = stack[-1]
    if fr.node.hit:
        fr.cont_ty = ContType.SUCCESS
        return stack
    fr.perm = rng.permutation(len(node.children))
    fr.i = 0
    return loop_step_st(stack, rng) 

def loop_step_st(stack, rng) -> list[Frame]:
    fr = stack[-1]
    if fr.i == len(fr.perm):
        return apply_fail_st(stack, rng)
    ch = node.children[fr.perm[fr.i]]
    child_frame = Frame(False, ContType.FAIL_LOOP_STEP, fr.node, fr.perm, fr.i+1)
    stack.append(child_frame)
    return search_sub_st(stack, rng)

def apply_fail_st(stack: list[Frame], rng: Generator) -> list[Frame]:
    fr = stack.pop()
    match fr.cont_ty:
        case ContType.FAIL_ROOT:
            return stack 
        case ContType.FAIL_LOOP_STEP:
            return loop_step_st(fr.node, fr.perm, fr.i, rng, stack)

def search_cps_st(node: Node, rng: Generator) -> Node|None:
    stack = [Frame(False, ContType.FAIL_ROOT, node, None, 0)]
    stack = search_sub_st(stack, rng)
    fr = stack.pop()
    return fr.node if fr.found else None


def main(seed: int):
    counts = np.array([3,4,7,4,5,8])
    max_depth = 4
    rng = np.random.default_rng(seed)

    root_ss = np.random.SeedSequence(seed)

    for i, ss in enumerate(root_ss.spawn(10000)):
        root = build_tree(np.random.default_rng(ss), counts, max_depth)
        found = search(root, np.random.default_rng(ss))
        found2 = search_loop(root, np.random.default_rng(ss))
        assert found == found2
        if i % 100 == 0:
            print(i, found, found2)


if __name__ == "__main__":
    fire.Fire(main)


