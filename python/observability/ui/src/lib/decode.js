// Foxglove channel → CDR decoder. foxglove_bridge advertises each ROS 2 topic
// with its full message definition (`schemaEncoding: "ros2msg"`), so the page
// decodes without knowing any message type in advance.
import { parse } from "@foxglove/rosmsg";
import { MessageReader } from "@foxglove/rosmsg2-serialization";

// Returns (bytes: ArrayBufferView) => object, or null for a channel this page
// cannot decode (another encoding, or an IDL schema).
export function makeDecoder(channel) {
  if (channel?.encoding !== "cdr") return null;
  if (channel.schemaEncoding && channel.schemaEncoding !== "ros2msg") return null;
  let reader;
  try {
    reader = new MessageReader(parse(String(channel.schema ?? ""), { ros2: true }));
  } catch {
    return null;
  }
  return (data) => reader.readMessage(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
}
