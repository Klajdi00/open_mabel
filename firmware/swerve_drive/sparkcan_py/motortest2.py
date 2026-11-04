#!/usr/bin/env python3
import os
import sys
import time
import struct

# Import the minimal API from your package
from sparkcan_py import SparkFlex

# Try to import optional enums/methods if your bindings include them
try:
    from sparkcan_py import IdleMode
except Exception:
    IdleMode = None

JS_EVENT_BUTTON = 0x01
JS_EVENT_AXIS   = 0x02
JS_EVENT_INIT   = 0x80

def open_joystick(path="/dev/input/js0"):
    # Non-blocking open so our control loop never stalls
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    return fd

def read_js_event(fd):
    """Read one joystick event if available. Return (type, number, value) or None if no data."""
    try:
        data = os.read(fd, 8)
        if len(data) != 8:
            return None
        t, value, etype, number = struct.unpack("IhBB", data)
        etype = etype & ~JS_EVENT_INIT  # strip init bit
        return etype, number, value
    except BlockingIOError:
        return None

def clamp(x, lo, hi):
    return lo if x < lo else hi if x > hi else x

def main():
    # Create motors (IDs 1..8) with short spacing like the C++ code
    try:
        motorPID  = SparkFlex("can0", 1)
        time.sleep(0.003)
        motorPID2 = SparkFlex("can0", 2)
        time.sleep(0.003)
        motorPID3 = SparkFlex("can0", 3)
        time.sleep(0.003)
        motorPID4 = SparkFlex("can0", 4)
        time.sleep(0.003)
        motor   = SparkFlex("can0", 5)
        time.sleep(0.003)
        motor1  = SparkFlex("can0", 6)
        time.sleep(0.003)
        motor2  = SparkFlex("can0", 7)
        time.sleep(0.003)
        motor3  = SparkFlex("can0", 8)
        time.sleep(0.003)
    except Exception as e:
        print(f"Failed to create SparkFlex controllers: {e}", file=sys.stderr)
        return 1

    # Optional: basic config if available in your bindings
    def config_pid(m):
        # Only run if methods exist
        if hasattr(m, "ClearStickyFaults"): m.ClearStickyFaults()
        if IdleMode and hasattr(m, "SetIdleMode"): m.SetIdleMode(IdleMode.kCoast)
        if hasattr(m, "SetP"): m.SetP(0, 0.2)
        if hasattr(m, "SetI"): m.SetI(0, 0.0)
        if hasattr(m, "SetD"): m.SetD(0, 0.1)
        if hasattr(m, "BurnFlash"): m.BurnFlash()

    for m in (motorPID, motorPID2, motorPID3, motorPID4):
        config_pid(m)

    if hasattr(motorPID, "BurnFlash"):
        print("Flash burned for motors ID 1, 2, 3, and 4, waiting 5 seconds...")
        time.sleep(5.0)
        print("Motors ID 1, 2, 3, and 4 ready for operation.")

    # Joystick
    try:
        jsfd = open_joystick("/dev/input/js0")
        print("Joystick opened successfully at /dev/input/js0")
    except Exception as e:
        print(f"Could not open joystick: {e}", file=sys.stderr)
        return 1

    desired_position = 0.0
    desired_velocity = 0.0  # used for PID motors (IDs 1..4)

    # Absolute encoder position offsets for (IDs 5..8)
    motor_offset  = 0.25
    motor1_offset = 0.0
    motor2_offset = 0.25
    motor3_offset = 0.0

    time.sleep(5.0)  # optional initial wait like C++

    print("Use left stick X-axis to control position (0.0 .. 1.0)")
    print("Use right stick Y-axis to control speed (-4.0 .. +4.0 units)")
    print("Ctrl+C to exit")

    start = time.monotonic()
    loop_period = 0.003  # ~3 ms like the C++ sleep_between

    try:
        while (time.monotonic() - start) < 100.0:
            # Heartbeat all motors
            motorPID.Heartbeat()
            motorPID2.Heartbeat()
            motorPID3.Heartbeat()
            motorPID4.Heartbeat()
            motor.Heartbeat()
            motor1.Heartbeat()
            motor2.Heartbeat()
            motor3.Heartbeat()

            # Read any joystick event
            ev = read_js_event(jsfd)
            if ev:
                etype, number, value = ev
                if etype == JS_EVENT_AXIS and number == 0:
                    # Left stick X → position [0,1]
                    joystick = float(value) / 32767.0  # [-1, 1]
                    desired_position = (joystick + 1.0) * 0.5
                    desired_position = clamp(desired_position, 0.0, 1.0)
                elif etype == JS_EVENT_AXIS and number == 3:
                    # Right stick Y → velocity [-4, 4] (negative for “up = forward” like C++)
                    joystick = float(value) / 32767.0  # [-1, 1]
                    desired_velocity = clamp(-joystick * 4.0, -4.0, 4.0)

            # Apply velocity to the four PID motors
            motorPID.SetVelocity(desired_velocity)
            motorPID2.SetVelocity(desired_velocity)
            motorPID3.SetVelocity(desired_velocity)
            motorPID4.SetVelocity(desired_velocity)

            # Apply position (with offsets) to the other four motors
            motor.SetPosition(desired_position + motor_offset)
            motor1.SetPosition(desired_position + motor1_offset)
            motor2.SetPosition(desired_position + motor2_offset)
            motor3.SetPosition(desired_position + motor3_offset)

            # Print status
            try:
                v1 = motorPID.GetVelocity()
                v2 = motorPID2.GetVelocity()
                v3 = motorPID3.GetVelocity()
                v4 = motorPID4.GetVelocity()
                pos = motor.GetAbsoluteEncoderPosition()
            except Exception:
                # If any getter isn’t bound, skip printing
                v1 = v2 = v3 = v4 = pos = float('nan')

            sys.stdout.write(
                f"\rPID Target={desired_velocity:5.2f} | "
                f"ID1={v1:6.2f} | ID2={v2:6.2f} | ID3={v3:6.2f} | ID4={v4:6.2f} | "
                f"POS={pos:6.2f} Deg"
            )
            sys.stdout.flush()

            # Pace the loop
            time.sleep(loop_period)

    except KeyboardInterrupt:
        print("\nStopping...")

    finally:
        try:
            os.close(jsfd)
        except Exception:
            pass
        print("\nDone.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
