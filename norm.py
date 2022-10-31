import math

vectorsFile = open("test.txt", "r")

vectors = vectorsFile.readlines()


for vector in vectors:
    sum = 0
    num = 0
    for value in vector.split():
        num = float(value)
        sum = sum + (num*num)
    print(math.sqrt(sum))
