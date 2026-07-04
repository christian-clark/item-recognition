import math
import sys

# header
sys.stdin.readline()

print("serialPosition,surpReduction")
for l in sys.stdin:
    items = l.strip().split()
    names = items[0].split(',')
    listlen = len(names)
    min_surp = -math.log(listlen)
    serial_position = int(items[1])
    surp0 = float(items[2])
    surp = float(items[3])
    assert min_surp < surp0
    increase = (surp0 - surp) / (surp0 - min_surp)
    print("{},{}".format(serial_position, increase))
