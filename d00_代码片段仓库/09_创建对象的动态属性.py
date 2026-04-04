# 定义了类MyUser，并实现里面的方法
class MyUser:
    def __init__(self, value12: dict):
        self.value = value12

    def get_class_to_id(self, dict12: dict):
        self.value = dict12
        return self.value


# 继承MyUser类, 调用父类的方法
class MyTest(MyUser):
    def __init__(self, dict13: dict):
        super(MyTest, self).__init__(dict13)
        class_to_id = self.get_class_to_id(dict13)
        self.class_to_id = class_to_id


# 直接定义类及其方法
class MyTest1:
    def __init__(self, dict14: dict):
        self.class_to_id = dict14


dict1 = {1: 'zhou', 2: 'wen', 3: 'jing'}

value = MyTest1(dict1)
# Python 允许你给一个对象动态添加属性 id_to_class是一个动态属性
value.id_to_class = {v: k for k, v in value.class_to_id.items()}
print(value.id_to_class)
