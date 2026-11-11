import assert from "node:assert/strict";
import test from "node:test";
import { parse } from "@foxglove/rosmsg";
import { MessageWriter } from "@foxglove/rosmsg2-serialization";
import { makeDecoder } from "./decode.js";
import {
  applyMarkers,
  cloudToPoints,
  compose,
  invert,
  occupancyToRGBA,
  rotate,
  sceneObjects,
  TfBuffer,
  wantedTopics,
} from "./scene.js";

// Message definitions exactly as foxglove_bridge advertises them (ros2msg, with
// dependencies appended after "MSG:" separators).
const SEP = "\n================================================================================\nMSG: ";
const TIME = "builtin_interfaces/Time\nint32 sec\nuint32 nanosec";
const HEADER = `std_msgs/Header\nbuiltin_interfaces/Time stamp\nstring frame_id${SEP}${TIME}`;
const POSE = "geometry_msgs/Pose\nPoint position\nQuaternion orientation" +
  `${SEP}geometry_msgs/Point\nfloat64 x\nfloat64 y\nfloat64 z` +
  `${SEP}geometry_msgs/Quaternion\nfloat64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1`;

const SCHEMAS = {
  "sensor_msgs/msg/PointCloud2":
    "std_msgs/Header header\nuint32 height\nuint32 width\nPointField[] fields\nbool is_bigendian\n" +
    "uint32 point_step\nuint32 row_step\nuint8[] data\nbool is_dense" +
    `${SEP}${HEADER}${SEP}sensor_msgs/PointField\nuint8 INT8=1\nuint8 FLOAT32=7\nstring name\nuint32 offset\nuint8 datatype\nuint32 count`,
  "nav_msgs/msg/OccupancyGrid":
    "std_msgs/Header header\nMapMetaData info\nint8[] data" +
    `${SEP}${HEADER}${SEP}nav_msgs/MapMetaData\nbuiltin_interfaces/Time map_load_time\nfloat32 resolution\nuint32 width\nuint32 height\ngeometry_msgs/Pose origin` +
    `${SEP}${POSE}`,
  "tf2_msgs/msg/TFMessage":
    "geometry_msgs/TransformStamped[] transforms" +
    `${SEP}geometry_msgs/TransformStamped\nstd_msgs/Header header\nstring child_frame_id\nTransform transform` +
    `${SEP}${HEADER}${SEP}geometry_msgs/Transform\nVector3 translation\nQuaternion rotation` +
    `${SEP}geometry_msgs/Vector3\nfloat64 x\nfloat64 y\nfloat64 z${SEP}geometry_msgs/Quaternion\nfloat64 x 0\nfloat64 y 0\nfloat64 z 0\nfloat64 w 1`,
};

// Encode with the real CDR writer, decode through the page's decoder.
function roundTrip(schemaName, msg) {
  const schema = SCHEMAS[schemaName];
  const bytes = new MessageWriter(parse(schema, { ros2: true })).writeMessage(msg);
  const decode = makeDecoder({ encoding: "cdr", schemaEncoding: "ros2msg", schemaName, schema });
  assert.ok(decode, schemaName);
  return decode(bytes);
}

const header = (frame_id) => ({ stamp: { sec: 1, nanosec: 0 }, frame_id });
const near = (a, b, eps = 1e-6) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < eps, `${a} ≉ ${b}`));

test("the decoder refuses what it can't read", () => {
  assert.equal(makeDecoder({ encoding: "json" }), null);
  assert.equal(makeDecoder({ encoding: "cdr", schemaEncoding: "ros2idl", schema: "" }), null);
});

test("transforms compose and invert", () => {
  const yaw90 = { t: [1, 0, 0], q: [0, 0, Math.SQRT1_2, Math.SQRT1_2] };
  near(rotate(yaw90.q, [1, 0, 0]), [0, 1, 0]);
  const back = compose(invert(yaw90), yaw90);
  near(back.t, [0, 0, 0]);
  near(compose(yaw90, { t: [1, 0, 0], q: [0, 0, 0, 1] }).t, [1, 1, 0]);
});

test("a decoded /tf tree answers lookups between connected frames only", () => {
  const msg = roundTrip("tf2_msgs/msg/TFMessage", {
    transforms: [
      { header: header("world"), child_frame_id: "base", transform: { translation: { x: 1, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: Math.SQRT1_2, w: Math.SQRT1_2 } } },
      { header: header("base"), child_frame_id: "gripper", transform: { translation: { x: 0.5, y: 0, z: 0.2 }, rotation: { x: 0, y: 0, z: 0, w: 1 } } },
      { header: header("/other_root"), child_frame_id: "loose", transform: { translation: { x: 0, y: 0, z: 0 }, rotation: { x: 0, y: 0, z: 0, w: 1 } } },
    ],
  });
  const tf = new TfBuffer();
  tf.addMessage(msg);
  near(tf.lookup("world", "gripper").t, [1, 0.5, 0.2]);
  near(tf.lookup("base", "gripper").t, [0.5, 0, 0.2]);
  near(tf.lookup("gripper", "base").t, [-0.5, 0, -0.2]);
  assert.equal(tf.lookup("world", "loose"), null);
  assert.deepEqual(tf.frames(), ["base", "gripper", "loose", "other_root", "world"]);
});

test("an occupancy grid becomes a texture with unknown, free and occupied cells", () => {
  const msg = roundTrip("nav_msgs/msg/OccupancyGrid", {
    header: header("map"),
    info: {
      map_load_time: { sec: 0, nanosec: 0 },
      resolution: 0.025,
      width: 3,
      height: 1,
      origin: { position: { x: -1, y: -2, z: 0 }, orientation: { x: 0, y: 0, z: 0, w: 1 } },
    },
    data: new Int8Array([-1, 0, 100]),
  });
  const colors = { unknown: [1, 2, 3, 0], free: [200, 200, 200, 255], occupied: [0, 0, 0, 255] };
  const tex = occupancyToRGBA(msg, colors);
  assert.deepEqual([...tex.rgba], [1, 2, 3, 0, 200, 200, 200, 255, 0, 0, 0, 255]);
  assert.equal(tex.frame, "map");
  assert.ok(Math.abs(tex.resolution - 0.025) < 1e-6);
  near(tex.origin.t, [-1, -2, 0]);
  assert.equal(occupancyToRGBA({ info: { width: 2, height: 2 }, data: [0] }, colors), null);
});

test("a point cloud decodes xyz + packed rgb and respects the point budget", () => {
  const n = 10;
  const step = 16;
  const buf = new DataView(new ArrayBuffer(n * step));
  for (let i = 0; i < n; i++) {
    buf.setFloat32(i * step, i, true);
    buf.setFloat32(i * step + 4, -i, true);
    buf.setFloat32(i * step + 8, i === 3 ? Number.NaN : 0.5, true);
    buf.setUint32(i * step + 12, 0xff8000, true);
  }
  const msg = roundTrip("sensor_msgs/msg/PointCloud2", {
    header: header("/camera_link"),
    height: 1,
    width: n,
    fields: [
      { name: "x", offset: 0, datatype: 7, count: 1 },
      { name: "y", offset: 4, datatype: 7, count: 1 },
      { name: "z", offset: 8, datatype: 7, count: 1 },
      { name: "rgb", offset: 12, datatype: 7, count: 1 },
    ],
    is_bigendian: false,
    point_step: step,
    row_step: step * n,
    data: new Uint8Array(buf.buffer),
    is_dense: false,
  });
  const all = cloudToPoints(msg);
  assert.equal(all.count, 9); // the NaN point is skipped
  assert.equal(all.frame, "camera_link");
  near([...all.positions.subarray(0, 6)], [0, 0, 0.5, 1, -1, 0.5]);
  near([...all.colors.subarray(0, 3)], [1, 128 / 255, 0]);
  const some = cloudToPoints(msg, 4);
  assert.ok(some.decimated && some.count <= 4);
});

test("marker arrays add, replace and delete primitives", () => {
  const store = new Map();
  const cube = { ns: "static", id: 1, type: 1, action: 0, header: header("base"), pose: { position: { x: 0.3, y: 0.1, z: 0 } }, scale: { x: 0.2, y: 0.3, z: 0.02 }, color: { r: 1, g: 0.5, b: 0, a: 0.4 } };
  applyMarkers(store, { markers: [cube, { ...cube, id: 2, type: 99 }] });
  assert.equal(store.size, 1); // unknown type ignored
  const m = store.get("static/1");
  assert.equal(m.kind, "cube");
  near(m.pose.t, [0.3, 0.1, 0]);
  assert.deepEqual(m.color, [1, 0.5, 0, 0.4]);
  applyMarkers(store, { markers: [{ ...cube, action: 2 }] });
  assert.equal(store.size, 0);
  applyMarkers(store, { markers: [cube, { action: 3 }] });
  assert.equal(store.size, 0);
});

test("a TRIANGLE_LIST marker (a declared mesh fixture) is kept with its points", () => {
  const store = new Map();
  const tri = { ns: "static", id: 7, type: 11, action: 0, header: header("base"), scale: { x: 1, y: 1, z: 1 }, color: { r: 1, g: 0.5, b: 0, a: 0.4 }, points: [{ x: 0, y: 0, z: 0 }, { x: 1, y: 0, z: 0 }, { x: 0, y: 1, z: 0 }] };
  applyMarkers(store, { markers: [tri] });
  const m = store.get("static/7");
  assert.equal(m.kind, "triangles");
  assert.deepEqual(m.points, [[0, 0, 0], [1, 0, 0], [0, 1, 0]]);
});

test("scene objects keep only finite positions and report their age", () => {
  const objs = sceneObjects(
    {
      objects: [
        { id: "eraser_1", label: "eraser", x: 0.18, y: -0.07, z: 0.01, frame_id: "base", confidence: 0.94, last_seen_ns: 9e9 },
        { label: "ghost", x: null, y: 0, z: 0 },
      ],
    },
    10e9,
  );
  assert.equal(objs.length, 1);
  assert.deepEqual([objs[0].label, objs[0].frame, objs[0].ageS], ["eraser", "base", 1]);
});

test("only the layers that are on are subscribed", () => {
  const advertised = ["/tf", "/tf_static", "/map", "/joint_states", "/robot_description", "/openral/robot_description", "/octomap_point_cloud_centers", "/openral/viz/static_geometry", "/openral/viz/collisions"];
  const on = { robot: true, grid: false, cloud: true, collision: true, payload: true };
  assert.deepEqual(
    [...wantedTopics(advertised, on)].sort(),
    ["/joint_states", "/octomap_point_cloud_centers", "/openral/robot_description", "/openral/viz/collisions", "/openral/viz/static_geometry", "/tf", "/tf_static"],
  );
  assert.deepEqual([...wantedTopics(advertised, {})].sort(), ["/tf", "/tf_static"]);
});
