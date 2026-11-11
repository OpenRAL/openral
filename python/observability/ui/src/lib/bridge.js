// The 3D scene's link to the robot's foxglove_bridge (through the robot's /ws
// proxy). Subscribes only to the topics asked for; keeps the newest raw message
// of snapshot topics and decodes it when the scene draws (`take`), while
// incremental topics (TF, markers, the URDF) are decoded and delivered in order.
import { FoxgloveClient } from "@foxglove/ws-protocol";
import { makeDecoder } from "./decode.js";

const INCREMENTAL = new Set([
  "tf2_msgs/msg/TFMessage",
  "visualization_msgs/msg/MarkerArray",
  "std_msgs/msg/String",
]);

// foxglove_bridge 3.x (built on the Foxglove SDK) accepts only `foxglove.sdk.v1` and
// answers HTTP 400 to a client that offers just the legacy name; older bridges know
// only the legacy one. Same wire format for everything this page uses, so offer both.
export const SUBPROTOCOLS = ["foxglove.sdk.v1", FoxgloveClient.SUPPORTED_SUBPROTOCOL];

// Mesh downloads share the socket with live data: a few at a time, each with its
// own deadline counted from when it is sent, not from when it was queued.
const ASSET_PARALLEL = 3;
const ASSET_TIMEOUT_MS = 30000;

export class Bridge {
  // onEvent(topic, msg): incremental messages, decoded, in arrival order.
  // onTopics(topics[]): the advertised topic list changed.
  // onStatus("open" | "closed", detail?), never from inside the constructor;
  // "closed" exactly once per connection.
  // WebSocketImpl: the WebSocket constructor (tests pass their own).
  constructor({ url, onEvent, onTopics, onStatus, WebSocketImpl = globalThis.WebSocket }) {
    this.handlers = { onEvent, onTopics, onStatus };
    this.channels = new Map(); // topic -> channel
    this.decoders = new Map(); // channel id -> decode fn | null
    this.subs = new Map(); // topic -> subscription id
    this.subTopic = new Map(); // subscription id -> topic
    this.latest = new Map(); // topic -> { data, at }
    this.assets = new Map(); // request id -> { resolve, reject, timer }
    this.assetQueue = []; // waiting to be sent: { uri, resolve, reject }
    this.nextAsset = 1;
    this.wanted = new Set();
    this.isClosed = false;
    const ws = new WebSocketImpl(url, SUBPROTOCOLS);
    this.client = new FoxgloveClient({ ws });
    this.client.on("open", () => onStatus?.("open"));
    this.client.on("close", (e) => this.#closed(e?.reason || `closed (${e?.code ?? "?"})`));
    // "error" also fires for one frame the client cannot parse on a healthy socket,
    // and a real failure is followed by "close": only "close" ends the connection.
    this.client.on("error", () => {});
    this.client.on("advertise", (chs) => {
      for (const ch of chs) this.channels.set(ch.topic, ch);
      this.#sync();
      onTopics?.([...this.channels.keys()]);
    });
    this.client.on("unadvertise", (ids) => {
      for (const [topic, ch] of this.channels) {
        if (!ids.includes(ch.id)) continue;
        this.channels.delete(topic);
        this.latest.delete(topic); // a re-advertised topic may carry another schema
        const sid = this.subs.get(topic);
        if (sid != null) this.subTopic.delete(sid);
        this.subs.delete(topic);
      }
      onTopics?.([...this.channels.keys()]);
    });
    this.client.on("message", (m) => this.#message(m));
    this.client.on("fetchAssetResponse", (r) => {
      const p = this.assets.get(r.requestId);
      if (!p) return;
      this.assets.delete(r.requestId);
      clearTimeout(p.timer);
      if (r.status === 0) p.resolve(r.data);
      else p.reject(new Error(r.error || "asset refused"));
      this.#pumpAssets();
    });
  }

  // Subscribe to exactly these topics (those advertised now or later).
  want(topics) {
    this.wanted = new Set(topics);
    this.#sync();
  }

  // The newest message on a snapshot topic, decoded, once: null when nothing new.
  take(topic) {
    const m = this.latest.get(topic);
    if (!m) return null;
    this.latest.delete(topic);
    const ch = this.channels.get(topic);
    const decode = ch && this.#decoder(ch);
    try {
      return decode ? decode(m.data) : null;
    } catch {
      return null; // a message that doesn't match its schema: skip it
    }
  }

  topics() {
    return [...this.channels.keys()];
  }

  // A mesh by URI (package://…), through the bridge's `assets` capability.
  fetchAsset(uri) {
    return new Promise((resolve, reject) => {
      if (this.isClosed) return reject(new Error("bridge closed"));
      this.assetQueue.push({ uri, resolve, reject });
      this.#pumpAssets();
    });
  }

  #pumpAssets() {
    while (this.assets.size < ASSET_PARALLEL && this.assetQueue.length) {
      const { uri, resolve, reject } = this.assetQueue.shift();
      const id = this.nextAsset++;
      const timer = setTimeout(() => {
        this.assets.delete(id);
        reject(new Error(`asset timed out: ${uri}`));
        this.#pumpAssets();
      }, ASSET_TIMEOUT_MS);
      this.assets.set(id, { resolve, reject, timer });
      try {
        this.client.fetchAsset(uri, id);
      } catch (e) {
        this.assets.delete(id);
        clearTimeout(timer);
        reject(e);
      }
    }
  }

  close() {
    this.closedByUs = true;
    this.client.close();
  }

  #closed(detail) {
    if (this.isClosed) return;
    this.isClosed = true;
    for (const p of this.assets.values()) {
      clearTimeout(p.timer);
      p.reject(new Error("bridge closed"));
    }
    this.assets.clear();
    for (const q of this.assetQueue.splice(0)) q.reject(new Error("bridge closed"));
    this.handlers.onStatus?.("closed", this.closedByUs ? null : detail);
  }

  #decoder(ch) {
    if (!this.decoders.has(ch.id)) this.decoders.set(ch.id, makeDecoder(ch));
    return this.decoders.get(ch.id);
  }

  #sync() {
    for (const [topic, sid] of this.subs) {
      if (this.wanted.has(topic)) continue;
      try {
        this.client.unsubscribe(sid);
      } catch {
        // socket closing
      }
      this.subs.delete(topic);
      this.subTopic.delete(sid);
      this.latest.delete(topic);
    }
    for (const topic of this.wanted) {
      const ch = this.channels.get(topic);
      if (!ch || this.subs.has(topic) || !this.#decoder(ch)) continue;
      try {
        const sid = this.client.subscribe(ch.id);
        this.subs.set(topic, sid);
        this.subTopic.set(sid, topic);
      } catch {
        // not open yet: the next advertise retries
      }
    }
  }

  #message({ subscriptionId, data }) {
    const topic = this.subTopic.get(subscriptionId);
    if (!topic) return;
    const ch = this.channels.get(topic);
    if (ch && INCREMENTAL.has(ch.schemaName)) {
      try {
        this.handlers.onEvent?.(topic, this.#decoder(ch)(data));
      } catch {
        // undecodable: skip
      }
      return;
    }
    this.latest.set(topic, { data });
  }
}
