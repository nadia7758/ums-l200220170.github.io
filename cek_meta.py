from metaflow import FlowSpec, step

class HelloFlow(FlowSpec):
    @step
    def start(self):
        print("Metaflow is connected!")
        self.next(self.end)

    @step
    def end(self):
        print("Flow completed!")

if __name__ == '__main__':
    HelloFlow()

