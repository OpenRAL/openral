"""Send one ExecuteRskill goal for the scripted driver (direct dispatch) and record its result.

Usage: send_goal.py <result.json> [deadline_s]
"""

import json
import sys
import time

import rclpy
from openral_msgs.action import ExecuteRskill
from rclpy.action import ActionClient
from rclpy.node import Node

out = sys.argv[1]
deadline = float(sys.argv[2]) if len(sys.argv) > 2 else 300.0
rclpy.init()
node = Node("pick_place_goal_client")
client = ActionClient(node, ExecuteRskill, "/openral/execute_rskill")
if not client.wait_for_server(timeout_sec=30.0):
    sys.exit("no /openral/execute_rskill server")
goal = ExecuteRskill.Goal()
goal.rskill_id = "local/isaac-pick-place-scripted"
goal.prompt = "pick up the objects one at a time and put each down on the support"
goal.deadline_s = deadline
fut = client.send_goal_async(goal)
rclpy.spin_until_future_complete(node, fut, timeout_sec=60.0)
handle = fut.result()
rec = {"t": time.time(), "accepted": bool(handle and handle.accepted)}
if handle and handle.accepted:
    res_fut = handle.get_result_async()
    rclpy.spin_until_future_complete(node, res_fut, timeout_sec=deadline + 120.0)
    r = res_fut.result()
    if r is not None:
        rec.update(
            t_done=time.time(),
            success=r.result.success,
            failure_kind=r.result.failure_kind,
            failure_reason=r.result.failure_reason,
            trace_id=r.result.trace_id,
            status=r.status,
        )
with open(out, "w") as fh:
    json.dump(rec, fh, indent=1)
print(json.dumps(rec), flush=True)
