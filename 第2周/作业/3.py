lis = input().split()
n = int(lis[0])
m = int(lis[1])
names = set(input().split())
for i in range(m):
    students = input().split()
    for name in students:
        if name in names:
            names.discard(name)
print(len(names))