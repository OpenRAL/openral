"""openral Sensors — sensor adapters, catalog and launch generators.

Public surface:
- ``CATALOG``: global :class:`SensorCatalog` populated by every vendor module
  on import.  Used by ``openral sensor list / show`` and the (future) ``catalog:``
  reference field on ``RobotDescription`` sensors.
- ``SensorCatalog``, ``SensorCatalogEntry``: the registry types.
- Per-vendor factories — see each ``<vendor>.py`` module:
    * ``realsense``: D415 / D435 / D435i bundles.
    * ``luxonis``:   OAK-D Pro bundle.
    * ``usb_uvc``:   Logitech C920.
    * ``force_torque``: Robotiq FT 300-S 6-axis F/T sensor.
- Launch helpers (RealSense): ``bundle_to_node_params``, ``generate_launch_py``,
  ``calibrate_camera_cmd``.
"""

# Import every vendor module so the global CATALOG is populated on
# `import openral_sensors`.  Order does not matter; each module registers
# its own entries with replace=True.
from openral_sensors import (  # reason: side-effect import populates CATALOG
    force_torque,
    luxonis,
    usb_uvc,
)
from openral_sensors.catalog import (
    CATALOG,
    SensorCatalog,
    SensorCatalogEntry,
    SensorSignature,
)
from openral_sensors.luxonis import oak_d_pro_bundle
from openral_sensors.realsense import (
    bundle_to_node_params,
    calibrate_camera_cmd,
    generate_launch_py,
    realsense_d415_bundle,
    realsense_d435_bundle,
    realsense_d435i_bundle,
)

__all__ = [
    "CATALOG",
    "SensorCatalog",
    "SensorCatalogEntry",
    "SensorSignature",
    "bundle_to_node_params",
    "calibrate_camera_cmd",
    "generate_launch_py",
    "oak_d_pro_bundle",
    "realsense_d415_bundle",
    "realsense_d435_bundle",
    "realsense_d435i_bundle",
]
__version__ = "0.1.0"
