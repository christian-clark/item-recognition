import math
import sys

# header
sys.stdin.readline()

print("serialPosition,surpReduction")
for l in sys.stdin:
    items = l.strip().split()
    names = items[0].split(',')
    listlen = len(names)
    min_surp = 0
    serial_position = int(items[2])
    # convert surprisal to probability
    surp0 = float(items[3])
    surp = float(items[4])
    assert min_surp < surp0
    increase = (surp0 - surp) / (surp0 - min_surp)
    print("{},{}".format(serial_position, increase))
