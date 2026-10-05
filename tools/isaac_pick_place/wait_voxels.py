"""Block until /openral/world_voxels has delivered N messages (default 30) or 900 s pass."""

import sys
import time

import rclpy
from openral_msgs.msg import OccupancyVoxels
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

need = int(sys.argv[1]) if len(sys.argv) > 1 else 30
rclpy.init()
node = Node("pick_place_wait_voxels")
got = [0]
node.create_subscription(
    OccupancyVoxels,
    "/openral/world_voxels",
    lambda _m: got.__setitem__(0, got[0] + 1),
    QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE, depth=1
    ),
)
t0 = time.time()
while got[0] < need and time.time() - t0 < 900:
    rclpy.spin_once(node, timeout_sec=0.5)
print("voxels", got[0], "after", round(time.time() - t0, 1), "s")
sys.exit(0 if got[0] >= need else 1)
