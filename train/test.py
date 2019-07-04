import numpy

s = [1,2,3,4]

n = numpy.array(s)

q = n[numpy.newaxis, numpy.newaxis, ]

print(q)