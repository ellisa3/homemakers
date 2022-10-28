import math

vectorsFile = open("test.txt", "r")

vectors = vectorsFile.readlines()

sum = 0
for vector in vectors:
    for value in vector.split():
        num = float(value)
        sum = sum + (num*num)
    print(math.sqrt(sum))

print(math.sqrt(sum))
