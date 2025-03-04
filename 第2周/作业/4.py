n = int(input())
lis = []
for i in range(n):
    temp_lis = []
    count = int(input())
    for j in range(count):
        temp_lis.append(input().split())
    lis.append(temp_lis)


final_lis = [ele for e in lis for ele in e]
sorted_final_lis = sorted(final_lis, key = lambda x : int(x[1]), reverse = True)

for i in range(len(sorted_final_lis)):
    print(*sorted_final_lis[i])