"""Solves an instance.

Modify this file to implement your own solvers.

For usage, run `python3 solve.py --help`.
"""
from sklearn.cluster import KMeans
import numpy as np

import argparse
from pathlib import Path
import sys
from typing import Callable, Dict
from point import Point
from instance import Instance
from solution import Solution
from file_wrappers import StdinFileWrapper, StdoutFileWrapper

def solve_naive(instance: Instance) -> Solution:
    return Solution(
        instance=instance,
        towers=instance.cities,
    )

def kmeans(instance: Instance) -> Solution:
    data = np.array(instance.cities_list)
    i = len(instance.cities)
    while True:
        kmeans = KMeans(i, init='k-means++', n_init=20).fit(data)
        towers = [Point(int(center[0]), int(center[1])) for center in kmeans.cluster_centers_]
        s = Solution(
        instance=instance,
        towers=towers
        )
        if s.valid():
            ans = towers
            i =  i - 1
        else: break
    # kmeans = KMeans(120, init='k-means++', n_init=20).fit(data)
    # towers = [Point(int(center[0]), int(center[1])) for center in kmeans.cluster_centers_]
    return Solution(
        instance=instance,
        towers=ans
    )

SOLVERS: Dict[str, Callable[[Instance], Solution]] = {
    "naive": solve_naive,
    "kmeans": kmeans
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
