def test_import_without_ros() -> None:
    from openral_hal.sim_sensor_bridge import SimSensorBridge

    assert SimSensorBridge.__name__ == "SimSensorBridge"
