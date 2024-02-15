# 부모 클래스
class Animal:
    def __init__(a, name, sound):
        a.name = name
        a.sound = sound

    def make_sound(a):
        print(f"{a.name} 가 {a.sound} 소리를 냅니다.")

# 자식 클래스
class Dog(Animal):  # Animal 클래스를 상속받음
    def wag_tail(a):
        print(f"{a.name} 가 꼬리를 흔듭니다.")

class Cat(Animal):  # Animal 클래스를 상속받음
    def scratch(a):
        print(f"{a.name} 가 할퀴기를 합니다.")

# 각 클래스의 인스턴스 생성
dog = Dog("멍멍이", "왈왈")
cat = Cat("야옹이", "야옹")

# 부모 클래스의 메서드 호출
dog.make_sound()  # 멍멍이 가 왈왈 소리를 냅니다.
cat.make_sound()  # 야옹이 가 야옹 소리를 냅니다.

# 자식 클래스의 메서드 호출
dog.wag_tail()    # 멍멍이 가 꼬리를 흔듭니다.
cat.scratch()     # 야옹이 가 할퀴기를 합니다.
