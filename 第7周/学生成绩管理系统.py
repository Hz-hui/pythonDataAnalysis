class InvalidGradeError(Exception):
    def __init__(self,grade):
        self.grade = grade
        self.message = f"成绩无效:{grade}"
     
class Student:
    def __init__(self, name):
        self.name = name
        self.grades = []

    def add_grade(self,grade):
        if grade <= 100 and grade >= 0:
            self.grades.append(grade)
        else:
            raise InvalidGradeError(grade)
            

    def total_score(self):
        return sum(self.grades)

    def average_score(self):
        if len(self.grades) == 0:
            return 0
        else:
            return self.total_score()/len(self.grades)

if __name__ == "__main__":
    #输入学生姓名并且创建类实例student
    name = input()
    student = Student(name)

    #输入学生成绩
    grades = map(int, input().split())

    #添加成绩
    for grade in grades:
        try:
            student.add_grade(grade)
        except InvalidGradeError as ige:
            print(ige.message)

    print(f"总成绩:{student.total_score()},平均成绩:{student.average_score():.1f}")