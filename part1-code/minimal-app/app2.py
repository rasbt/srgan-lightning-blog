import lightning as L


class WordComponent1(L.LightningWork):
    def __init__(self, word):
        super().__init__(run_once=False, parallel=True)
        self.word = word

    def run(self):
        print(self.word)


class MyRootComponent(L.LightningFlow):
    def __init__(self) -> None:
        super().__init__()
        self.hello = WordComponent1("hello")
        self.world = WordComponent1("world")

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
