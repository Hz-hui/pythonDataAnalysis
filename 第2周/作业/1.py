students = eval(input())
sorted_students = sorted(students, key = lambda x : x[2], reverse = True)
for i in range(len(students)):
    print(*sorted_students[i])