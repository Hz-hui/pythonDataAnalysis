class A:
    #类变量，所有实例共享
    ID = 0
    def __init__(self):
        self._id=A.ID#通过类对象引用类变量
        A.ID += 1 
        self.ID = 10000#这是哪个ID?实例变量，不是类变量，不建议重名

    def print_id(self):
        print(f'{self} with id = {self._id}')

    def print_ID(self):
        print(self.ID)


a_list = [A() for i in range(100)]
for a in a_list:
    a.print_id()
for a in a_list:
    print(a.ID)#通过实例对象访问类变量
for a in a_list:
	A.print_id(a)#通过类对象调用实例方法
for a in a_list:
    a.print_ID()