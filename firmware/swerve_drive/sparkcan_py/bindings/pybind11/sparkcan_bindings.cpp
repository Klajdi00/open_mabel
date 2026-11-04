// bindings/pybind11/sparkcan_bindings.cpp
#include <pybind11/pybind11.h>
#include <string>
#include <cstdint>

#include "SparkBase.hpp"
#include "SparkFlex.hpp"
#include "SparkMax.hpp"

namespace py = pybind11;
using guard = py::call_guard<py::gil_scoped_release>;

PYBIND11_MODULE(_sparkcan, m) {
    m.doc() = "Minimal Python bindings for SparkCAN: SetPosition, SetVelocity, GetAbsoluteEncoderPosition, GetVelocity, Heartbeat";

    // Control-type + idle-mode enums (parity with the YOR robot's setup: drive
    // motors need their ctrl type set or SetVelocity is ignored; steer motors
    // need kPosition so the controller closes the loop on the absolute encoder).
    py::enum_<CtrlType>(m, "CtrlType")
        .value("kDutyCycle", CtrlType::kDutyCycle)
        .value("kVelocity", CtrlType::kVelocity)
        .value("kVoltage", CtrlType::kVoltage)
        .value("kPosition", CtrlType::kPosition)
        .value("kSmartMotion", CtrlType::kSmartMotion)
        .value("kSmartVelocity", CtrlType::kSmartVelocity);

    py::enum_<IdleMode>(m, "IdleMode")
        .value("kCoast", IdleMode::kCoast)
        .value("kBrake", IdleMode::kBrake);

    // Bind just the requested SparkBase APIs so SparkFlex/Max inherit them.
    py::class_<SparkBase>(m, "SparkBase")
        .def("SetPosition", &SparkBase::SetPosition, py::arg("position"), guard())
        .def("SetVelocity", &SparkBase::SetVelocity, py::arg("velocity"), guard())
        // Open-loop zero command used to safely idle a motor (limp/coast) without
        // the velocity PID fighting stiction. Critical for safe GUI startup.
        .def("SetDutyCycle", &SparkBase::SetDutyCycle, py::arg("dutyCycle"), guard())
        .def("ResetFaults", &SparkBase::ResetFaults, guard())
        .def("GetAbsoluteEncoderPosition", &SparkBase::GetAbsoluteEncoderPosition, guard())
        .def("GetVelocity", &SparkBase::GetVelocity, guard())
        // Extra telemetry getters for the Swerve Studio GUI (temp / current /
        // voltage / primary position). Already implemented in SparkBase; the GUI
        // probes for them with hasattr() so it degrades gracefully if an older
        // build of this extension is installed.
        .def("GetPosition", &SparkBase::GetPosition, guard())
        .def("GetTemperature", &SparkBase::GetTemperature, guard())
        .def("GetCurrent", &SparkBase::GetCurrent, guard())
        .def("GetVoltage", &SparkBase::GetVoltage, guard())
        .def("GetDutyCycle", &SparkBase::GetDutyCycle, guard())
        .def("GetAbsoluteEncoderVelocity", &SparkBase::GetAbsoluteEncoderVelocity, guard())
        .def("ClearStickyFaults", &SparkBase::ClearStickyFaults, guard())
        // Closed-loop control type + idle mode (applied on connect, like YOR).
        .def("SetCtrlType", &SparkBase::SetCtrlType, py::arg("type"), guard())
        .def("SetIdleMode", &SparkBase::SetIdleMode, py::arg("mode"), guard())
        // Periodic status-frame period setters (milliseconds). The controllers
        // do NOT broadcast frame 2 (position+velocity) or frame 5 (absolute
        // encoder) until these are enabled, which is why velocity and the steer
        // encoder read blank. The GUI enables them on connect.
        .def("SetPeriodicStatus0Period", &SparkBase::SetPeriodicStatus0Period, py::arg("period"), guard())
        .def("SetPeriodicStatus1Period", &SparkBase::SetPeriodicStatus1Period, py::arg("period"), guard())
        .def("SetPeriodicStatus2Period", &SparkBase::SetPeriodicStatus2Period, py::arg("period"), guard())
        .def("SetPeriodicStatus3Period", &SparkBase::SetPeriodicStatus3Period, py::arg("period"), guard())
        .def("SetPeriodicStatus4Period", &SparkBase::SetPeriodicStatus4Period, py::arg("period"), guard())
        .def("SetPeriodicStatus5Period", &SparkBase::SetPeriodicStatus5Period, py::arg("period"), guard())
        .def("Heartbeat", &SparkBase::Heartbeat, guard())
        .def("heartbeat", &SparkBase::Heartbeat, guard());  // alias

    // Concrete devices (inherit SparkBase methods)
    py::class_<SparkFlex, SparkBase>(m, "SparkFlex")
        .def(py::init<const std::string&, uint8_t>(),
             py::arg("interface"), py::arg("device_id"));

    py::class_<SparkMax, SparkBase>(m, "SparkMax")
        .def(py::init<const std::string&, uint8_t>(),
             py::arg("interface"), py::arg("device_id"));
}
