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
        self.hello = WordComponent1("Hello")
        self.world = WordComponent1("World")
        self.counter = 0

    def run(self):
        self.counter += 1
        if self.counter <= 6:
            print("I just go with the Flow!")
        self.hello.run()
        self.world.run()


app = L.LightningApp(MyRootComponent())

# run this app via the following command:
# lightning run app app.py
