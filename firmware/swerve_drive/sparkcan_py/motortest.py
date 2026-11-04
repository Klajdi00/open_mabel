from sparkcan_py import SparkFlex
import time
m = SparkFlex("can0", 1)
m.Heartbeat()
m.SetVelocity(0.0)
while True:
    m.Heartbeat()
    if m.GetVelocity()>0.001:
        m.SetVelocity(0.5)
    print("Vel:", m.GetVelocity())
    time.sleep(0.005)
