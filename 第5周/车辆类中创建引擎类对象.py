class Engine():
    def __init__(self, power):
        self.power = power

class Vehicle():
    def __init__(self, wheels):
        self.wheels = 4
        self.engine = engine.power
    def display_info(self):
        print(f"轮子数量:{self.wheels}")
        print(f"发动机动力:{self.engine}")

power = int(input())

engine = Engine(power)

vehicle = Vehicle(4)

vehicle.display_info()  # 打印车辆信息