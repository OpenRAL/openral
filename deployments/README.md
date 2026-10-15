# `deployments/` — retired

Real deploys now use `DeployScene` YAMLs from `scenes/deploy/`.

Use:

```bash
openral deploy run --config scenes/deploy/<workcell>.yaml
```

Robot facts (joints, sensors, rates, limits) belong in
`robots/<robot_id>/robot.yaml`; a host's robot-camera bindings belong in its
unit overlay (`robots/<robot_id>/units/<unit>.yaml`, selected by `robot_unit`).
Workcell facts (the active robot, scene, workspace cameras, the host-specific
`hal:` binding such as a serial port, safety tightening, and additive allowed
collision pairs) belong in the `DeployScene`. HAL precedence is
`--hal` > scene `hal:` > `robot.yaml`.
