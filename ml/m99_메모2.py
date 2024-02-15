# 부모 클래스 (상위 클래스)
class Animal:
    def __init__(self, name):
        self.name = name

    def speak(self):
        raise NotImplementedError("Subclass must implement abstract method")


# 자식 클래스 (하위 클래스)
class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"


class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"


# 자식 클래스의 인스턴스 생성
dog = Dog("Buddy")
cat = Cat("Whiskers")

# 각 동물이 소리를 내도록 함
print(dog.speak())  # 출력: Buddy says Woof!
print(cat.speak())  # 출력: Whiskers says Meow!
