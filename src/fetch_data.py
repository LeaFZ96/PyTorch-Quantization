import sys

print(sys.argv[1])

fin = open(sys.argv[1])
fout = open('./output.txt', 'w')

line = fin.readline()
nums = []

while (line):
    if '*' in line:
        nums.append(line.split()[2])
    line = fin.readline()

fin.close()

for i in nums:
    fout.write(i)
    fout.write(', ')

fout.close()