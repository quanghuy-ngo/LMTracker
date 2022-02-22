#!/usr/bin/python
import random
import sys

total_all = 1051430459
total_fail = 12840308
total_success = total_all - total_fail

# numbers of rows wanted and seed
wanted = 400000
seed = 2

# number of output files
nfiles = 30

skip_success = total_success / (wanted / 2.0)
skip_fail = total_fail / (wanted / 2.0)

print >>sys.stderr, 'Taking 1/%.1f success, 1/%.1f fail' % (
    skip_success, skip_fail)

# skip header
sys.stdin.readline()

outfiles = []
counts = []
for i in range(nfiles):
    filename = 'md/msample%d.csv' % i
    print 'File %d is %s' % (i, filename)
    outfiles.append(open(filename, 'w'))
    counts.append([0, 0])

# Create reproducable random
r = random.Random()
r.seed(seed)

# Split the file
for line in sys.stdin:
    failed = ',Fail' in line
    if failed:
        idx = int(r.random() * skip_fail)
    else:
        idx = int(r.random() * skip_success)
    if idx < nfiles:
        outfiles[idx].write(line)
        counts[idx][failed] += 1

# print stats
for i in range(nfiles):
    outfiles[i].close()
    print 'File %d: %d success + %d fail = %d' % (
        i, counts[i][1], counts[i][0], sum(counts[i]))
