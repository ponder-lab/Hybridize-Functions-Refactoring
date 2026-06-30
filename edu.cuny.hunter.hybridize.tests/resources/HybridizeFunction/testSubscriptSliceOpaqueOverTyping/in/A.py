import argparse


def check(value):
    return float(value) < 1.0


def plain(value):
    return float(value) < 1.0


parser = argparse.ArgumentParser()
parser.add_argument("--xs", nargs="+", type=str)
args = parser.parse_args()

for x in args.xs[1::2]:
    check(x)

for y in args.xs:
    plain(y)
