"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""

import argparse
from pathlib import Path
import sys
from typing import Callable, Dict
from point import Point
from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper

from collections import namedtuple
from itertools import product
from math import sqrt
from pprint import pprint as pp


def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )


Pt = namedtuple('Pt', 'x, y')
Cir = namedtuple('Cir', 'x, y, r')

def circles_from_p1p2r(p1, p2, r):
    (x1, y1), (x2, y2) = p1, p2
    x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
    if p1 == p2:
        #raise ValueError('coincident points gives infinite number of Circles')
        return None, None
    # delta x, delta y between points
    dx, dy = x2 - x1, y2 - y1
    # dist between points
    q = sqrt(dx**2 + dy**2)
    if q > 2.0*r:
        #raise ValueError('separation of points > diameter')
        # One answer
        c1 = Cir(x = x1,
                y = y1,
                r = abs(r))
        # The other answer
        c2 = Cir(x = x2,
                y = y2,
                r = abs(r))
        return c1, c2
    # halfway point
    x3, y3 = (x1+x2)/2, (y1+y2)/2
    # distance along the mirror line
    d = sqrt(r**2-(q/2)**2)
    # One answer
    c1 = Cir(x = x3 - d*dy/q,
             y = y3 + d*dx/q,
             r = abs(r))
    # The other answer
    c2 = Cir(x = x3 + d*dy/q,
             y = y3 - d*dx/q,
             r = abs(r))
    return c1, c2

def covers(c, pt):
    return (c.x - int(pt[0]))**2 + (c.y - int(pt[1]))**2 <= c.r**2

"""
def method(instance: Instance) -> Solution:
    r, points = 3, instance.cities_list
    n, p = len(points), points  
    # All circles between two points (which can both be the same point)
    circles = set(sum([[c1, c2]
                        for c1, c2 in [circles_from_p1p2r(p1, p2, r) for p1, p2 in product(p, p)]
                        if c1 is not None], []))
    # points covered by each circle 
    coverage = {c: {pt for pt in points if covers(c, pt)}
                for c in circles}
    # Ignore all but one of circles covering points covered in whole by other circles
    #print('\nwas considering %i circles' % len(coverage))
    items = sorted(coverage.items(), key=lambda keyval:len(keyval[1]))
    for i, (ci, coveri) in enumerate(items):
        for j in range(i+1, len(items)):
            cj, coverj = items[j]
            if not coverj - coveri:
                coverage[cj] = {}
    coverage = {key: val for key, val in coverage.items() if val}
    #print('Reduced to %i circles for consideration' % len(coverage))

    # Greedy coverage choice
    chosen, covered = [], set()
    while len(covered) < n:
        _, nxt_circle, nxt_cov = max((len(pts - covered), c, pts)
                                        for c, pts in coverage.items())
        delta = nxt_cov - covered
        covered |= nxt_cov
        chosen.append([nxt_circle, delta])

    # Output
    print('\n%i points' % n)
    pp(points)
    print('A minimum of circles of radius %g to cover the points (And the extra points they covered)' % r)
    pp(chosen)
    """
   

def method(instance: Instance) -> Solution:
    #for r, points in [(3, [Pt(*i) for i in [(1, 3), (0, 2), (4, 5), (2, 4), (0, 3)]]),
    #                  (2, [Pt(*i) for i in [(1, 3), (0, 2), (4, 5), (2, 4), (0, 3)]]),
    #                  (3, [Pt(*i) for i in [(-5, 5), (-4, 4), (3, 2), (1, -1), (-3, 2), (4, -2), (6, -6)]])]:
    r, points = 3, [Pt(*i) for i in [(0, 0), (14, 5), (2, 7), (3, 19), (27, 17), (11, 11), (14, 29), 
            (7, 11), (5, 29), (4, 3), (17, 18), (29, 29), (13, 1), (20, 25), (19, 6)]]
    #r, points = 3, instance.cities_tuples
    #r, points = 3, [Pt(*i) for i in [(1, 11), (0, 2), (4, 5), (2, 4), (0, 3)]]
    n, p = len(points), points  
    # All circles between two points (which can both be the same point)
    circles = set(sum([[c1, c2]
                        for c1, c2 in [circles_from_p1p2r(p1, p2, r) for p1, p2 in product(p, p)]
                        if c1 is not None], []))
    # points covered by each circle 
    coverage = {c: {pt for pt in points if covers(c, pt)}
                for c in circles}
    # Ignore all but one of circles covering points covered in whole by other circles
    #print('\nwas considering %i circles' % len(coverage))
    items = sorted(coverage.items(), key=lambda keyval:len(keyval[1]))
    for i, (ci, coveri) in enumerate(items):
        for j in range(i+1, len(items)):
            cj, coverj = items[j]
            if not coverj - coveri:
                coverage[cj] = {}
    coverage = {key: val for key, val in coverage.items() if val}
    #print('Reduced to %i circles for consideration' % len(coverage))

    # Greedy coverage choice
    chosen, covered = [], set()
    while len(covered) < n:
        _, nxt_circle, nxt_cov = max((len(pts - covered), c, pts)
                                        for c, pts in coverage.items())
        delta = nxt_cov - covered
        covered |= nxt_cov
        chosen.append([nxt_circle, delta])

    towers = [Point(circ[0].x, circ[0].y) for circ in chosen]
    # Output
    #print('\n%i points' % n)
    #pp(points)
    #print('A minimum of circles of radius %g to cover the points (And the extra points they covered)' % r)
    #pp(chosen)
    pp(towers)

    return Solution(
        instance=instance,
        towers=towers
    )



SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "x": method
}


# You shouldn't need to modify anything below this line.
def infile(args):
    if args.input == "-":
        return StdinFileWrapper()

    return Path(args.input).open("r")

def outfile(args):
    if args.output == "-":
        return StdoutFileWrapper()

    return Path(args.output).open("w")

def main(args):
    with infile(args) as f:
        instance = Instance.parse(f.readlines())
        solver = SOLVERS[args.solver]
        solution = solver(instance)
        assert solution.valid()
        with outfile(args) as g:
            print("# Penalty: ", solution.penalty(), file=g)
            solution.serialize(g)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve a problem instance.")
    parser.add_argument("input", type=str, help="The input instance file to "
                        "read an instance from. Use - for stdin.")
    parser.add_argument("--solver", required=True, type=str,
                        help="The solver type.", choices=SOLVERS.keys())
    parser.add_argument("output", type=str, 
                        help="The output file. Use - for stdout.", 
                        default="-")
    main(parser.parse_args())
