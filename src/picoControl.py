import picod
import time

class PicoControl:
    def __init__(self, com, relayPin):
        self.pico = picod.pico(com)
        self.relayPin = relayPin
        self.pico.reset()
        self.pico.gpio_open(gpio=self.relayPin)
        #self.pico.GPIO_set_dir(out_GPIO=self.relayPin, out_LEVEL=0)
        self.pico.gpio_set_output(self.relayPin, level=0)
        self.pico.gpio_set_pull(self.relayPin, pull=picod.PULL_DOWN)
        if not self.pico.connected:
            exit()

    def setIRCutFilter(self, state):
        self.pico.gpio_write(self.relayPin, not state)
        return
    
if __name__ == "__main__":
    pico = PicoControl("COM3", 0)
    pico.setIRCutFilter(1)
    time.sleep(10)
    pico.setIRCutFilter(0)
