class Student:
    def __init__(self, name, *args, **kwargs):
        self.name = name
        self.other_attributes = args  # args是一个元组，包含额外的位置参数
        self.additional_info = kwargs  # kwargs是一个字典，包含额外的关键字参数

# 创建Student类的实例，使用*args和**kwargs传递参数
student1 = Student("Charlie", "Computer Science", "Interesting", element=121, year=2023, major="Engineering")
print(student1.name)           # 输出: Charlie
print(student1.other_attributes) # 输出: ('Computer Science',)
print(student1.additional_info) # 输出: {'year': 2023, 'major': 'Engineering'}