import picod
import time

class PicoControl:
    def __init__(self, com, relayPin):
        self.pico = picod.pico(com)
        self.relayPin = relayPin
        self.pico.reset()
        self.pico.GPIO_open(self.relayPin)
        #self.pico.GPIO_set_dir(out_GPIO=self.relayPin, out_LEVEL=0)
        if not self.pico.connected:
            exit()

    def setIRCutFilter(self, state):
        self.pico.gpio_write(self.relayPin, state)
        return
    
if __name__ == "__main__":
    pico = PicoControl("COM3", 0)
    pico.setIRCutFilter(1)
    time.sleep(2)
    pico.setIRCutFilter(0)
    time.sleep(2)
    pico.setIRCutFilter(1)
    time.sleep(2)
    pico.setIRCutFilter(0)
