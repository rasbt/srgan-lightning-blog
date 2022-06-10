import lightning as L


class WordComponent(L.LightningWork):
    def __init__(self, word):
        super().__init__()
        self.word = word

    def run(self):
        print(self.word)


class MyRootComponent(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.hello = WordComponent("hello")
        self.world = WordComponent("world")

        self.counter = 0

    def run(self):
        self.counter += 1
        if self.counter < 6:
            print("Hey, I am swimming in the LightningFlow!")
        self.hello.run()
        self.world.run()


app = L.LightningApp(MyRootComponent())

# run as
# lightning run app app.py
