import lightning as L


class WordComponent(L.LightningWork):
    def __init__(self, word):
        super().__init__(parallel=True, cache_calls=False)
        self.word = word

    def run(self):
        print(self.word)


class MyRootComponent(L.LightningFlow):
    def __init__(self):
        super().__init__()
        self.hello = WordComponent("Hello")
        self.world = WordComponent("World")
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
