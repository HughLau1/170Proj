"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""
#from sklearn.cluster import KMeans
#import numpy as np

import argparse
import collections
from pathlib import Path
import sys
from typing import Callable, Dict
import numpy as np

from sklearn.cluster import KMeans

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
        self.x = int(x)
        self.y = int(y)
    def __repr__(self):
            return ', '.join((str(self.x),str(self.y)))

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

    newx1 = x3 - d*dy/q
    newy1 = y3 + d*dx/q

    newx2 = x3 + d*dy/q
    newy2 = y3 - d*dx/q


    if newx1 < newx2:
        newx1 = int(newx1) + 1
        newx2 = int(newx2)

    elif newx1 > newx2:
        newx1 = int(newx1)
        newx2 = int(newx2) + 1
    else:
        pass
    
    if newy1 < newy2:
        newy1 = int(newy1) + 1
        newy2 = int(newy2)

    elif newy1 > newy2:
        newy1 = int(newy1)
        newy2 = int(newy2) + 1

    else:
        pass
    
    if 0 <= newx1 < boundary and 0 <= newx2 < boundary and 0 <= newy1 < boundary and 0 <= newy2 < boundary:
        return [CircleObj(newx1, newy1), CircleObj(newx2, newy2)]
    else: return 
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

    print("coverage len = " + str(len(coverage)))

    coverage_sorted_by_len = sorted(coverage.items(), key = lambda keyvalpair:len(keyvalpair[1]))
    
    for i, (ci, coveri) in enumerate(coverage_sorted_by_len):
        for j in range(i+1, len(coverage_sorted_by_len)):
            cj, coverj = coverage_sorted_by_len[j]
            if not coverj - coveri:
                coverage[cj] = {}
    coverage = {key: val for key, val in coverage.items() if val}
    print("coverage len after removing= " + str(len(coverage)))
    
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
