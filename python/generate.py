"""Generates instance inputs of small, medium, and large sizes.

Modify this file to generate your own problem instances.

For usage, run `python3 generate.py --help`.
"""

import argparse
from pathlib import Path
from typing import Callable, Dict

from instance import Instance
from size import Size
from point import Point
from file_wrappers import StdoutFileWrapper


def make_small_instance() -> Instance:
    """Creates a small problem instance.

    Size.SMALL.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
"""
    # cities=[
    #     Point(x=0, y=0),
    #     Point(x=14, y=5),
    #     Point(x=2, y=7),
    #     Point(x=3, y=19),
    #     Point(x=27, y=17),
    #     Point(x=11, y=11),
    #     Point(x=14, y=29),
    #     Point(x=7, y=11),
    #     Point(x=5, y=29),
    #     Point(x=4, y=3),
    #     Point(x=17, y=18),
    #     Point(x=28, y=29),
    #     Point(x=13, y=1),
    #     Point(x=20, y=25),
    #     Point(x=19, y=6),
    # ]

    lines = """
# Small instance.
15
30
3
8
0 0 
14 5
2 7
3 19
27 17
11 11 
14 29
7 11
5 29
4 3
17 18
29 29
13 1
20 25
19 6
    """.strip().splitlines()
    # YOUR CODE HERE
    # return Size.SMALL.instance(cities)
    i = Instance.parse(lines)
    return i


def make_medium_instance() -> Instance:
    """Creates a medium problem instance.

    Size.MEDIUM.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
    """
    # cities = []
    # # YOUR CODE HERE
    # return Size.MEDIUM.instance(cities)
    
    lines = """
# Medium instance.
50
50
3
10
25 34
45 29
42 46
28 16
43 18
29 34
25 9
37 11
13 35
38 33
13 28
2 28
47 8
21 38
47 44
5 16
14 38
17 46
1 17
40 0
25 37
9 31
21 8
39 45
21 30
32 10
27 30
47 5
32 11
13 9
48 44
16 6
39 10
36 33
42 44
20 21
3 11
31 31
22 18
7 49
38 49
25 20
41 9
44 37
9 36
37 39
13 34
32 14
8 10
15 6
    """.strip().splitlines()
    return Instance.parse(lines)
    # YOUR CODE HERE
    # return Size.SMALL.instance(cities)


def make_large_instance() -> Instance:
    """Creates a large problem instance.

    Size.LARGE.instance() handles setting instance constants. Your task is to
    specify which cities are in the instance by constructing Point() objects,
    and add them to the cities array. The skeleton will check that the instance
    is valid.
    """
    # cities = []
    # # YOUR CODE HERE
    # return Size.LARGE.instance(cities)
    lines = """
# Large instance.
200
100
3
14
46 67
49 81
43 68
97 7
84 33
86 93
33 59
1 72
24 6
9 29
46 57
89 43
23 95
1 32
66 48
51 34
86 18
83 24
82 91
40 50
12 15
5 41
19 62
21 29
26 1
41 72
74 27
99 46
60 81
19 48
4 56
6 84
51 73
33 7
55 44
64 98
6 5
12 96
19 25
3 8
29 60
26 91
1 14
23 4
57 5
82 95
21 25
79 86
92 85
46 22
35 27
55 33
16 45
71 67
28 88
81 65
14 63
43 26
39 21
97 66
72 33
84 85
98 87
38 86
6 61
3 33
55 67
85 32
22 41
21 23
18 9
83 11
46 62
11 77
22 95
74 83
30 71
25 91
96 95
52 34
75 63
18 92
51 71
80 44
22 42
63 10
68 76
0 33
13 78
44 50
37 52
9 13
31 42
86 11
63 54
81 8
1 31
67 53
37 85
96 62
90 71
90 44
61 17
81 12
25 59
80 5
32 99
65 45
94 52
94 10
95 93
97 32
9 39
40 0
33 70
24 29
16 57
70 36
27 2
71 97
92 16
40 83
23 48
93 72
28 49
5 76
21 33
39 49
12 22
30 76
88 6
30 59
77 46
17 30
13 29
48 79
15 57
48 85
36 77
56 47
85 95
35 12
72 58
39 53
65 61
74 2
78 15
47 93
3 2
1 78
91 90
70 14
80 51
76 17
96 68
63 89
18 49
80 17
48 2
56 50
42 60
83 99
11 14
97 53
2 56
87 42
74 39
48 72
60 8
45 80
30 8
64 19
82 53
35 10
45 68
52 30
23 50
89 41
41 49
97 25
64 63
35 97
76 92
52 36
10 80
49 92
3 75
66 9
66 60
19 99
39 82
68 94
23 49
27 49
85 22
81 49
80 81
20 66
33 77
41 93
    """.strip().splitlines()
    return Instance.parse(lines)

# You shouldn't need to modify anything below this line.
SMALL = 'small'
MEDIUM = 'medium'
LARGE = 'large'

SIZE_STR_TO_GENERATE: Dict[str, Callable[[], Instance]] = {
    SMALL: make_small_instance,
    MEDIUM: make_medium_instance,
    LARGE: make_large_instance,
}

SIZE_STR_TO_SIZE: Dict[str, Size] = {
    SMALL: Size.SMALL,
    MEDIUM: Size.MEDIUM,
    LARGE: Size.LARGE,
}

def outfile(args, size: str):
    if args.output_dir == "-":
        return StdoutFileWrapper()

    return (Path(args.output_dir) / f"{size}.in").open("w")


def main(args):
    for size, generate in SIZE_STR_TO_GENERATE.items():
        if size not in args.size:
            continue

        with outfile(args, size) as f:
            instance = generate()
            assert instance.valid(), f"{size.upper()} instance was not valid."
            assert SIZE_STR_TO_SIZE[size].instance_has_size(instance), \
                f"{size.upper()} instance did not meet size requirements."
            print(f"# {size.upper()} instance.", file=f)
            instance.serialize(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate problem instances.")
    parser.add_argument("output_dir", type=str, help="The output directory to "
                        "write generated files to. Use - for stdout.")
    parser.add_argument("--size", action='append', type=str,
                        help="The input sizes to generate. Defaults to "
                        "[small, medium, large].",
                        default=None,
                        choices=[SMALL, MEDIUM, LARGE])
    # action='append' with a default value appends new flags to the default,
    # instead of creating a new list. https://bugs.python.org/issue16399
    args = parser.parse_args()
    if args.size is None:
        args.size = [SMALL, MEDIUM, LARGE]
    main(args)
