class Shape:
	def draw(self):#抽象，规范，约定
		pass

class Triangle(Shape):
	def draw(self):
		print("draw a triangle")

class Rectangle(Shape):
	def draw(self):
		print("draw a rect")

class Circle(Shape):
	def draw(self):
		print("draw a circle")

def draw(shapes=None):
	if shapes:
		for s in shapes:
			s.draw()

shapes=[Triangle(),Rectangle(),Circle()]
draw(shapes)

class OddShape(Shape):
	def draw(self):
		print("draw an odd shape")


class Noshape:
	def draw(self):
		print("draw the no shape")


shapes=[Triangle(),Rectangle(),Circle(),OddShape(), Noshape()]
draw(shapes)