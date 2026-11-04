"""ros — the rclpy adapters: DDS topics in/out, mabel_hw underneath.

Each node module exposes ``build(cfg, hw=None, **_)``:
  * ``hw=`` (a shared :class:`mabel_hw.MabelHardware`, passed by the bridge
    supervisor) — the node routes commands through the facade's write-leased
    handles and reads its typed snapshots. This is the production shape.
  * ``hw=None`` (standalone: tests, single-subsystem debugging) — the node
    builds its own mabel_hw.subsystems core, exactly the objects the facade
    would own, and drives them directly.

The DDS contract (topic names, message types, QoS incl. latched torque_state)
is byte-identical to the pre-mabel_hw bridge — frozen by
tests/contract/contract.json. rclpy is allowed ONLY here and in
mabel_hw.cameras (enforced by tests/test_mabel_hw_pure.py).
"""
