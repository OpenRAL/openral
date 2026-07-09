# `deployments/` — retired

Real deploys now use `DeployScene` YAMLs from `scenes/deploy/`.

Use:

```bash
openral deploy run --config scenes/deploy/<workcell>.yaml
```

Robot facts such as serial ports, robot IPs, sensors, rates, and limits belong
in `robots/<robot_id>/robot.yaml`. Workcell facts such as the active robot,
scene, safety tightening, and additive allowed collision pairs belong in the
`DeployScene`.
