class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def move_to(self, dx, dy):
        self.x = self.x + dx
        self.y = self.y + dy
    def __str__(self):
        return f"({self.x},{self.y})"

x, y = map(int,input().split())
dx, dy = map(int, input().split())

p = Point(x, y)
print(p)

p.move_to(dx,dy)
print(p)