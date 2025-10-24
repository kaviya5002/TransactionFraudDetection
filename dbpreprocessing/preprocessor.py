a = 10
class preprocessor1():
    
    def __init__(self):
        self.a = 10
    def subtract(self):
        self.a -= 1
    def add(self):
        self.a += 1
    def printt(self):
        print(self.a)

    @staticmethod
    def add():
        global a
        a += 1
    @staticmethod
    def printt():
        global a
        print(a)
    
        
    
