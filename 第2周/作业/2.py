lis = input().split(',')
f_lis = [float(ele) for ele in lis]
sorted_lis = sorted(f_lis, reverse = True)
for i in range(len(f_lis)):
    print(sorted_lis[i])