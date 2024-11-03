import picod


class PicoControl:
    def __init__(self, relayPin):
        self.pico = picod.pico()
        self.relayPin = relayPin
        self.pico.reset()
        if not self.pico.connected:
            exit()

    def setIRCutFilter(self, state):
        self.pico.gpio_write(self.relayPin, state)
        return
