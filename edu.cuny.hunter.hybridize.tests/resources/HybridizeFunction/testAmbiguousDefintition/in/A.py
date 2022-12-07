class Test:
    def __init__(self, value):
        self.__value = value

    @property
    def value(self):
        return self.__value

    @value.setter
    def name(self, number):
        self.__value = number


if __name__ == "__main__":
    k = Test(1)
    print(k.value)  # using getter
    k.name = 2  # using setter
    print(k.value)  # using getter
