"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""
#from sklearn.cluster import KMeans
#import numpy as np

import argparse
from pathlib import Path
import sys
from typing import Callable, Dict

#from sklearn.cluster import KMeans
from point import Point
from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper

from collections import namedtuple
from itertools import product
from math import sqrt
from pprint import pprint as pp
#import numpy as np

def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

class PointObj:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __iter__(self):
        return (self.x, self.y)

class CircleObj:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)
        
    def __repr__(self):
            return ', '.join((str(self.x),str(self.y)))
        
    def __gt__ (self, other):
        # if self.x**2 + self.y**2 < other.x**2 + other.y**2:
        if self.x>other.x:
            return True
        else:
            return False
        
    def __eq__ (self, other):
        return self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))

def circles_from_p1p2r(p1, p2, boundary):
    x1, y1 = p1.x, p1.y
    x2, y2 = p2.x, p2.y
    if x1 == x2 and y1 == y2:
        return 
    
    dx = x2 - x1
    dy = y2 - y1
    q = sqrt(dx**2 + dy**2)
    
    if q > 2.0*3:
        return [CircleObj(x1, y1), CircleObj(x2, y2)]
    

    x3, y3 = (x1+x2)/2, (y1+y2)/2

    d = sqrt(3**2-(q/2)**2)

    if c1.x < c2.x:
        c1.x = int(c1.x) + 1
        c2.x = int(c2.x)
        #x1 = int(c1.x) + 1
        #x2 = int(c2.x)
    elif c1.x > c2.x:
        c1.x = int(c1.x)
        c2.x = int(c2.x) + 1
        #x1 = int(c1.x)
        #x2 = int(c2.x) + 1
    else:
        pass
    if c1.y < c2.y:
        c1.y = int(c1.y) + 1
        c2.y = int(c2.y)
        #y1 = int(c1.y) + 1
        #y2 = int(c2.y)
    elif c1.y > c2.y:
        c1.y = int(c1.y)
        c2.y = int(c2.y) + 1
        #y1 = int(c1.y)
        #y2 = int(c2.y) + 1
    else:
        pass
    #c1 = Cir(x = int(x1),
    #        y = int(y1),
    #        r = abs(r))
    #c2 = Cir(x = int(x2),
    #        y = int(y2),
    #        r = abs(r))
    return c1, c2

def covers(c, pt):
    return (c.x - int(pt.x))**2 + (c.y - int(pt.y))**2 <= 9
   

def method(instance: Instance) -> Solution:

    points = [PointObj(x,y) for x,y in instance.cities_list]
    n = len(points)
    print("n len = " + str(n))

    circles = sum([x for x in [circles_from_p1p2r(p1, p2, instance.grid_side_length) for p1, p2 in product(points, points)] if x], [])
    circles = set(circles)
    print("num of circles generated= " + str(len(circles)))
    # pp(circles)
    
    coverage = {c: {pt for pt in points if covers(c, pt)}
                for c in circles}
    # pp('coverage before')
    # pp(coverage)
    # Ignore all but one of circles covering points covered in whole by other circles
    #print('\nwas considering %i circles' % len(coverage))
    items = sorted(coverage.items(), key=lambda keyval:len(keyval[1]))
    for i, (ci, coveri) in enumerate(items):
        for j in range(i+1, len(items)):
            cj, coverj = items[j]
            if not coverj - coveri:
                coverage[cj] = {}
    #pp('coverage after')
    coverage = {key: val for key, val in coverage.items() if val}
    #print('Reduced to %i circles for consideration' % len(coverage))
    # pp(coverage)
    # Greedy coverage choice
    chosen, covered = [], set()
    while len(covered) < n:
        _, circ, pts = max((len(pts - covered), circ, pts) for circ, pts in coverage.items())
        pts_not_already_covered = pts - covered
        covered |= pts
        chosen.append([circ, pts_not_already_covered])
        
    towers = [Point(circ.x, circ.y) for circ, _ in chosen]
    print([pt.y for pt in towers])    

    return Solution(
        instance=instance,
        towers=towers
    )


def kmeans(instance: Instance) -> Solution:
    data = np.array(instance.cities_list)

    for i in range(1, len(data)):
        
        kmeans = KMeans(i, init='k-means++', n_init=20).fit(data)
        towers = [Point(int(center[0]), int(center[1])) for center in kmeans.cluster_centers_]
        s = Solution(instance=instance, towers=towers)
        if s.valid():
            break
    return s
    
SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "kmeans": kmeans, 
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
