class Singleton:
	_instance = None
	def __init__(self, name, volume):
		self.name=name#实例变量
		self.volume=volume#实例变量

	def __new__(cls,name,volume):#cls与self类似，代表将要构建的类型的名称
		if not Singleton._instance:#通过类对象引用类变量
		#if not hasattr(Singleton,'_instance'):
			Singleton._instance=object.__new__(cls)#cls实际上就是类名
			Singleton.__init__(Singleton._instance,name,volume)#self需要给实参
		return Singleton._instance

slist=[Singleton('z',100) for i in range(10)]
for s in slist:
	print(hex(id(s)),end='\t')
	print(f"{s.name}\t{s.volume}")