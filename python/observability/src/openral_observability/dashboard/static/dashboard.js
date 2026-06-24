(() => {
  const $ = (id) => document.getElementById(id);
  const num = (v, digits=2) => v == null ? "—" : Number(v).toFixed(digits);
  const fmtMs = (ms) => (ms == null || isNaN(ms) ? "—" : Number(ms).toFixed(1) + " ms");
  const fmtAge = (ts) => {
    if (!ts) return "—";
    const dt = Date.now() / 1000 - ts;
    if (dt < 1) return "now";
    if (dt < 60) return dt.toFixed(1) + "s";
    if (dt < 3600) return Math.floor(dt / 60) + "m";
    return Math.floor(dt / 3600) + "h";
  };
  const fmtTime = (ts) => ts ? new Date(ts * 1000).toTimeString().slice(0, 8) : "";

  function setId(el, value, kind) {
    el.classList.remove("accent", "info");
    if (value === undefined || value === null || value === "") {
      el.textContent = "—";
      el.classList.add("empty");
    } else {
      el.textContent = String(value);
      el.classList.remove("empty");
      if (kind === "accent") el.classList.add("accent");
      else if (kind === "info") el.classList.add("info");
    }
  }

  function renderIdentity(state) {
    const id = state.identity || {};
    setId($("id-service"), state.service_name);
    setId($("id-runmode"), state.run_mode, "accent");
    setId($("id-runid"), state.run_id ? state.run_id.slice(0, 12) : "");
    setId($("id-gitsha"), state.git_sha ? state.git_sha.slice(0, 8) : "");
    setId($("id-robot"), id["openral.hal.robot.model"]);
    setId($("id-hal"), id["openral.hal.adapter"]);
    setId($("id-ctrl"), id["openral.hal.control_mode"], "info");
    setId($("id-skill"), id["openral.rskill.id"] || id["rskill.id"]);
    setId($("id-skillrole"), id["openral.rskill.role"] || id["rskill.role"]);
    const eng = id["inference.engine"];
    const dev = id["inference.device"];
    setId($("id-engine"), eng || dev ? `${eng || "—"} · ${dev || "—"}` : "");
    setId($("id-horizon"), id["openral.rskill.action_horizon"] || id["openral.hal.action.horizon"]);
    setId($("id-kernel"), id["safety.kernel"]);
  }

  const PRIMARY_BY_FAMILY = {
    rskill_execute: ["rskill.id", "openral.skill.id"],
    inference: ["inference.engine", "rskill.id"],
    safety: ["safety.check_name", "openral.safety.check_name"],
  };
  const ATTR_KEYS_BY_FAMILY = {
    rskill_execute: ["rskill.id", "rskill.role", "skill.action_applied", "openral.tick.idx"],
    inference: ["inference.kind", "inference.chunk_index", "inference.chunk_size"],
    safety: ["safety.check_name", "safety.severity", "safety.clamped"],
  };
  const LATENCY_CLASS_BY_FAMILY = {
    rskill_execute: "",
    inference: "info",
    safety: "warn",
  };

  function renderCard(family, card) {
    const el = $("card-" + family);
    if (!el) return;
    if (!card) {
      el.classList.add("empty");
      el.querySelector(".primary").textContent = "waiting…";
      el.querySelector(".age").textContent = "—";
      const oldAttrs = el.querySelector(".attrs"); if (oldAttrs) oldAttrs.remove();
      const oldLat = el.querySelector(".latency"); if (oldLat) oldLat.remove();
      return;
    }
    el.classList.remove("empty");
    el.classList.toggle("error", card.status_code === 2);
    const keys = PRIMARY_BY_FAMILY[family] || [];
    let primary = card.name;
    for (const k of keys) {
      if (card.attrs && card.attrs[k] != null) { primary = String(card.attrs[k]); break; }
    }
    el.querySelector(".primary").textContent = primary;
    el.querySelector(".age").textContent = fmtAge(card.ts_unix);
    let lat = el.querySelector(".latency");
    if (!lat) {
      lat = document.createElement("div");
      lat.className = "latency";
      const latClass = LATENCY_CLASS_BY_FAMILY[family];
      if (latClass) lat.classList.add(latClass);
      el.insertBefore(lat, el.querySelector(".attrs") || null);
    }
    lat.textContent = fmtMs(card.duration_ms);
    let attrsEl = el.querySelector(".attrs");
    if (!attrsEl) {
      attrsEl = document.createElement("div"); attrsEl.className = "attrs";
      el.appendChild(attrsEl);
    }
    attrsEl.innerHTML = "";
    for (const k of (ATTR_KEYS_BY_FAMILY[family] || [])) {
      if (!card.attrs || card.attrs[k] == null) continue;
      const kEl = document.createElement("div"); kEl.className = "k"; kEl.textContent = k.split(".").slice(-1)[0];
      const vEl = document.createElement("div"); vEl.className = "v"; vEl.textContent = String(card.attrs[k]);
      attrsEl.appendChild(kEl); attrsEl.appendChild(vEl);
    }
  }

  function renderRobotState(rs, cmd) {
    const el = $("joints");
    $("robot-state-age").textContent = fmtAge(rs && rs.ts_unix);
    if (!rs || !rs.names || !rs.positions) {
      el.innerHTML = '<div class="empty-state">waiting for hal.read_state</div>';
      return;
    }
    el.innerHTML = "";
    const cmdNext = cmd && cmd.next_row;
    for (let i = 0; i < rs.names.length; i++) {
      const name = rs.names[i];
      const pos = rs.positions[i];
      const vel = rs.velocities ? rs.velocities[i] : null;
      const lo = (rs.limits_lo && rs.limits_lo[i] != null && rs.limits_lo[i] > -1e5) ? rs.limits_lo[i] : -Math.PI;
      const hi = (rs.limits_hi && rs.limits_hi[i] != null && rs.limits_hi[i] < 1e5) ? rs.limits_hi[i] : Math.PI;
      const span = (hi - lo) || 1;
      const dotPct = Math.max(0, Math.min(100, ((pos - lo) / span) * 100));
      const cmdPct = (cmdNext && cmdNext[i] != null)
        ? Math.max(0, Math.min(100, ((cmdNext[i] - lo) / span) * 100))
        : null;
      const velClass = vel != null && vel < 0 ? "neg" : "";
      const row = document.createElement("div");
      row.className = "joint-row";
      row.innerHTML = `
        <span class="name">${name}</span>
        <div class="bar">
          <div class="track"></div>
          ${cmdPct != null ? `<div class="cmd" style="left: ${cmdPct}%"></div>` : ""}
          <div class="dot" style="left: ${dotPct}%"></div>
        </div>
        <span class="val">${num(pos, 3)}</span>
        <span class="vel ${velClass}">${vel != null ? num(vel, 2) + " /s" : "—"}</span>
      `;
      el.appendChild(row);
    }
  }

  function renderCommands(cmd) {
    $("cmd-age").textContent = fmtAge(cmd && cmd.ts_unix);
    const el = $("cmd-attrs");
    el.innerHTML = "";
    if (!cmd || !cmd.next_row) {
      el.innerHTML = '<div class="empty-state">waiting for hal.send_action</div>';
      return;
    }
    const pairs = [
      ["control", cmd.control_mode || "—"],
      ["dim × horizon", `${cmd.dim || "?"} × ${cmd.horizon || "?"}`],
      ["applied", cmd.applied === false ? "✗ no" : "✓ yes"],
      ["next", "[" + cmd.next_row.slice(0, 6).map(v => num(v, 2)).join(", ") + (cmd.next_row.length > 6 ? ", …" : "") + "]"],
    ];
    if (cmd.gripper_position != null) pairs.push(["gripper", num(cmd.gripper_position, 2)]);
    if (cmd.gripper_force_n != null) pairs.push(["grip force", num(cmd.gripper_force_n, 2) + " N"]);
    for (const [k, v] of pairs) {
      const kEl = document.createElement("div"); kEl.className = "k"; kEl.textContent = k;
      const vEl = document.createElement("div"); vEl.className = "v"; vEl.textContent = v;
      el.appendChild(kEl); el.appendChild(vEl);
    }
  }

  function renderWorldState(ws) {
    $("ws-age").textContent = fmtAge(ws && ws.ts_unix);
    const attrs = $("ws-attrs");
    const diag = $("ws-diag");
    attrs.innerHTML = "";
    diag.innerHTML = "";
    if (!ws || ws.ts_unix == null) {
      attrs.innerHTML = '<div class="empty-state">waiting for world_state.snapshot</div>';
      return;
    }
    const pairs = [
      ["components stale", ws.components_stale ?? "—", false],
      ["latched error", ws.has_latched_error ? "✗ YES" : "✓ no", false],
      ["battery", ws.battery_pct != null ? num(ws.battery_pct, 1) + " %" : "—", ws.battery_pct == null],
    ];
    if (ws.ee_poses) {
      for (const [name, pose] of Object.entries(ws.ee_poses).slice(0, 3)) {
        if (pose && pose.length >= 3) {
          pairs.push([`ee ${name}`, `[${num(pose[0], 2)}, ${num(pose[1], 2)}, ${num(pose[2], 2)}]`, false]);
        }
      }
    }
    for (const [k, v, faint] of pairs) {
      const kEl = document.createElement("div"); kEl.className = "k"; kEl.textContent = k;
      const vEl = document.createElement("div"); vEl.className = "v" + (faint ? " faint" : ""); vEl.textContent = v;
      attrs.appendChild(kEl); attrs.appendChild(vEl);
    }
    if (ws.diagnostics) {
      for (const [k, v] of Object.entries(ws.diagnostics)) {
        const kEl = document.createElement("span"); kEl.className = "k"; kEl.textContent = k;
        const sev = String(v || "").toLowerCase();
        const pillClass = sev === "ok" ? "ok" : sev === "stale" ? "stale" : sev === "warn" ? "warn" : "error";
        const vEl = document.createElement("span"); vEl.className = "pill " + pillClass; vEl.textContent = sev || "?";
        diag.appendChild(kEl); diag.appendChild(vEl);
      }
    }
  }

  function renderPerception(perc) {
    const el = $("cameras");
    el.innerHTML = "";
    const cams = perc && perc.cameras;
    if (!cams || Object.keys(cams).length === 0) {
      el.innerHTML = '<div class="empty-state" style="grid-column: 1 / -1">waiting for sensors.read_latest</div>';
      return;
    }
    const entries = Object.entries(cams).sort((a, b) => a[0].localeCompare(b[0]));
    for (const [name, cam] of entries) {
      const isLive = cam.ts_unix && (Date.now() / 1000 - cam.ts_unix) < 2;
      const hasFrame = cam.thumbnail_jpeg_b64 || cam.width;
      const imgInner = hasFrame
        ? `<img src="/api/camera/${encodeURIComponent(name)}/stream" alt="${name}"
             onerror="this.replaceWith(Object.assign(document.createElement('div'),
               {className:'camera-placeholder',textContent:'stream unavailable'}))" />`
        : `<div class="camera-placeholder">no frames · ${cam.modality || "?"}</div>`;
      const ageMs = cam.age_ms != null ? num(cam.age_ms, 0) + " ms" : "—";
      const label = (cam.role || cam.modality || "cam").toString();
      const dims = `${cam.width || "?"} × ${cam.height || "?"} · ${cam.encoding || cam.modality || "?"}`;
      const fps = cam.fps != null ? num(cam.fps, 0) + " fps" : "";
      const div = document.createElement("div");
      div.className = "camera";
      div.innerHTML = `
        <div class="image-wrap">
          ${imgInner}
          <span class="corner tl"></span><span class="corner tr"></span>
          <span class="corner bl"></span><span class="corner br"></span>
          <div class="crosshair"></div>
          <div class="pill-row">
            ${isLive ? '<span class="pill">live</span>' : ''}
            <span class="pill neutral">${label}</span>
          </div>
        </div>
        <div class="footer">
          <span class="name">${name}</span>
          <span class="lat">${ageMs}</span>
          <span class="meta-sub">${dims}</span>
          <span class="meta-sub right">${fps}</span>
        </div>
      `;
      el.appendChild(div);
    }
  }

  // ADR-0025 — render the live 2D occupancy map. Mirrors the camera-card
  // pattern: empty-state when nothing has been emitted yet, switch to
  // an inline base64 PNG once the bridge sends a slam.occupancy_grid
  // span. Metadata (resolution / origin / frame_id / source node)
  // pinned below the image so operators can sanity-check what they
  // are looking at.
  // ADR-0025 — map world (metres) → map PNG pixel coords. The bridge
  // rasterises the OccupancyGrid with a vertical flip (PIL top-left vs
  // grid bottom-left), so pixel-y is mirrored: py = (height-1) - row.
  function worldToPixel(wx, wy, originX, originY, resolution, height) {
    const col = (wx - originX) / resolution;
    const row = (wy - originY) / resolution;
    return { px: col, py: (height - 1) - row };
  }

  // ADR-0025 — base-frame footprint vertices -> map pixel points. Rotate
  // each (bx,by) by yaw, translate to the robot's world pose, then reuse
  // worldToPixel (which applies the PNG vertical flip).
  function footprintToPixels(polygon, robotX, robotY, yaw, originX, originY, resolution, height) {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    return polygon.map(([bx, by]) => {
      const wx = robotX + bx * c - by * s;
      const wy = robotY + bx * s + by * c;
      return worldToPixel(wx, wy, originX, originY, resolution, height);
    });
  }

  function renderRobotMarker(slam) {
    const svg = $("slam-overlay");
    if (!svg) return;
    const W = slam.width, H = slam.height, res = slam.resolution_m;
    const haveCells = W && H && res;
    const havePose =
      slam.robot_x != null && slam.robot_y != null && slam.robot_yaw != null;
    if (!haveCells || !havePose) { svg.innerHTML = ""; return; }
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const ox = slam.origin_x || 0, oy = slam.origin_y || 0;
    const yaw = slam.robot_yaw;
    const { px, py } = worldToPixel(slam.robot_x, slam.robot_y, ox, oy, res, H);
    const accent = "var(--accent-world,#5cc6ff)";
    const poly = slam.footprint_polygon;

    if (Array.isArray(poly) && poly.length >= 3) {
      // Real base outline, oriented by yaw, + a heading line to the front edge.
      const pts = footprintToPixels(poly, slam.robot_x, slam.robot_y, yaw, ox, oy, res, H);
      const ptsStr = pts.map((p) => `${p.px},${p.py}`).join(" ");
      const frontDist = Math.max(...poly.map((p) => p[0]));
      const front = worldToPixel(
        slam.robot_x + frontDist * Math.cos(yaw),
        slam.robot_y + frontDist * Math.sin(yaw),
        ox, oy, res, H
      );
      const sw = Math.max(0.4, (frontDist / res) * 0.12);
      svg.innerHTML =
        `<polygon points="${ptsStr}" fill="${accent}" fill-opacity="0.25" ` +
        `stroke="${accent}" stroke-width="${sw}"/>` +
        `<line x1="${px}" y1="${py}" x2="${front.px}" y2="${front.py}" ` +
        `stroke="${accent}" stroke-width="${sw}" stroke-opacity="0.95" stroke-linecap="round"/>`;
      return;
    }

    // Fallback: footprint circle (real radius or fixed) + heading wedge.
    const rCells = slam.footprint_radius_m != null
      ? slam.footprint_radius_m / res
      : Math.max(4, 0.25 / res);
    // y flips in image space, so heading uses -sin(yaw).
    const hx = Math.cos(yaw), hy = -Math.sin(yaw);
    const tipX = px + hx * rCells * 1.6, tipY = py + hy * rCells * 1.6;
    const halfW = rCells * 0.55;
    const baseX = px + hx * rCells * 0.2, baseY = py + hy * rCells * 0.2;
    const leftX = baseX - hy * halfW, leftY = baseY + hx * halfW;
    const rightX = baseX + hy * halfW, rightY = baseY - hx * halfW;
    svg.innerHTML =
      `<circle cx="${px}" cy="${py}" r="${rCells}" fill="${accent}" ` +
      `fill-opacity="0.25" stroke="${accent}" stroke-width="${Math.max(0.5, rCells * 0.08)}"/>` +
      `<polygon points="${tipX},${tipY} ${leftX},${leftY} ${rightX},${rightY}" ` +
      `fill="${accent}" fill-opacity="0.9"/>`;
  }

  function renderSlamMap(slam) {
    const ageEl = $("slam-age");
    const empty = $("slam-empty");
    const wrap = $("slam-image-wrap");
    if (!slam || !slam.ts_unix || !slam.png_b64) {
      if (ageEl) ageEl.textContent = "—";
      if (empty) empty.style.display = "block";
      if (wrap) wrap.style.display = "none";
      const ov = $("slam-overlay"); if (ov) ov.innerHTML = "";
      return;
    }
    if (ageEl) ageEl.textContent = fmtAge(slam.ts_unix);
    if (empty) empty.style.display = "none";
    if (wrap) wrap.style.display = "block";
    const img = $("slam-image");
    if (img) img.src = "data:image/png;base64," + slam.png_b64;
    // Size the stage to the map's aspect ratio, scaled UP to fill the card (up
    // to a 340px height cap). Without this a small grid (e.g. 48×109) rendered
    // at native pixel size — a tiny thumbnail. The overlay SVG fills the same
    // stage so the robot/object markers stay aligned with the cells.
    const stage = $("slam-image-stage");
    if (stage && slam.width && slam.height) {
      const availW = (wrap && wrap.clientWidth) ? wrap.clientWidth : 480;
      const scale = Math.max(1, Math.min(availW / slam.width, 340 / slam.height));
      stage.style.width = Math.round(slam.width * scale) + "px";
      stage.style.height = Math.round(slam.height * scale) + "px";
    }
    const dims = $("slam-dims");
    if (dims) dims.textContent = (slam.width || "?") + " × " + (slam.height || "?") + " cells";
    const resEl = $("slam-resolution");
    if (resEl) {
      const r = slam.resolution_m;
      resEl.textContent = r != null ? "resolution " + num(r, 3) + " m/cell" : "";
    }
    const origin = $("slam-origin");
    if (origin) {
      origin.textContent = "origin (" + num(slam.origin_x || 0, 2) + ", " + num(slam.origin_y || 0, 2) + ")";
    }
    const frame = $("slam-frame");
    if (frame) frame.textContent = "frame " + (slam.frame_id || "?");
    const src = $("slam-source");
    if (src) src.textContent = "source " + (slam.source_node || "?");
    const poseEl = $("slam-pose");
    if (poseEl) {
      poseEl.textContent =
        slam.robot_x != null && slam.robot_y != null && slam.robot_yaw != null
          ? "robot (" + num(slam.robot_x, 2) + ", " + num(slam.robot_y, 2) +
            ") yaw " + num((slam.robot_yaw * 180) / Math.PI, 0) + "°"
          : "";
    }
    renderRobotMarker(slam);
  }

  // ADR-0030 — render the robot-perspective octomap pointcloud. Mirrors
  // renderSlamMap: empty-state until the first world.pointcloud span, then
  // an inline base64 PNG with n_points / range / frame / source pinned below.
  function renderWorldCloud(pc) {
    const ageEl = $("world-cloud-age");
    const empty = $("world-cloud-empty");
    const wrap = $("world-cloud-image-wrap");
    if (!pc || !pc.ts_unix || !pc.png_b64) {
      if (ageEl) ageEl.textContent = "—";
      if (empty) empty.style.display = "block";
      if (wrap) wrap.style.display = "none";
      return;
    }
    if (ageEl) ageEl.textContent = fmtAge(pc.ts_unix);
    if (empty) empty.style.display = "none";
    if (wrap) wrap.style.display = "block";
    const img = $("world-cloud-image");
    if (img) img.src = "data:image/png;base64," + pc.png_b64;
    const pts = $("world-cloud-points");
    if (pts) pts.textContent = (pc.n_points != null ? pc.n_points : "?") + " points";
    const range = $("world-cloud-range");
    if (range) range.textContent = pc.range_max_m != null ? "range " + num(pc.range_max_m, 1) + " m" : "";
    const frame = $("world-cloud-frame");
    if (frame) frame.textContent = "frame " + (pc.frame_id || "?");
    const src = $("world-cloud-source");
    if (src) src.textContent = "source " + (pc.source_node || "?");
  }

  // ADR-0038 — render the durable spatial-memory objects as a table. Empty
  // until the first world.scene_objects span (Reasoner preloaded map today;
  // World-State node once the perception object-lift producer lands). Rows are
  // built with textContent (labels are operator/perception controlled).
  function renderSceneObjects(so) {
    const ageEl = $("scene-objects-age");
    const empty = $("scene-objects-empty");
    const wrap = $("scene-objects-wrap");
    const objects = (so && Array.isArray(so.objects)) ? so.objects : [];
    if (!so || !so.ts_unix || objects.length === 0) {
      if (ageEl) ageEl.textContent = so && so.ts_unix ? fmtAge(so.ts_unix) : "—";
      if (empty) {
        empty.style.display = "block";
        empty.textContent = (so && so.ts_unix)
          ? "spatial memory is empty (0 objects remembered)"
          : "waiting for world.scene_objects (preload a scene graph via spatial_memory_path, or merge the perception object-lift producer)";
      }
      if (wrap) wrap.style.display = "none";
      return;
    }
    if (ageEl) ageEl.textContent = fmtAge(so.ts_unix);
    if (empty) empty.style.display = "none";
    if (wrap) wrap.style.display = "block";
    const rows = $("scene-objects-rows");
    if (rows) {
      rows.innerHTML = "";
      // Most-recently-seen first.
      const sorted = objects.slice().sort((a, b) => (b.last_seen_ns || 0) - (a.last_seen_ns || 0));
      for (const o of sorted) {
        const tr = document.createElement("tr");
        const label = document.createElement("td");
        label.style.padding = "2px 6px";
        label.textContent = (o.label || o.id || "?") + (o.is_container ? " ⬚" : "");
        const pos = document.createElement("td");
        pos.style.padding = "2px 6px";
        pos.textContent = "(" + num(o.x || 0, 2) + ", " + num(o.y || 0, 2) + ", " + num(o.z || 0, 2) + ")";
        const conf = document.createElement("td");
        conf.style.padding = "2px 6px";
        conf.textContent = o.confidence != null ? num(o.confidence, 2) : "—";
        const seen = document.createElement("td");
        seen.style.padding = "2px 6px";
        seen.textContent = o.last_seen_ns ? fmtAge(o.last_seen_ns / 1e9) : "—";
        const obs = document.createElement("td");
        obs.style.padding = "2px 6px";
        obs.textContent = o.observation_count != null ? o.observation_count : "—";
        tr.appendChild(label); tr.appendChild(pos); tr.appendChild(conf);
        tr.appendChild(seen); tr.appendChild(obs);
        rows.appendChild(tr);
      }
    }
    const countEl = $("scene-objects-count");
    if (countEl) countEl.textContent = objects.length + " object" + (objects.length === 1 ? "" : "s");
    const frameEl = $("scene-objects-frame");
    if (frameEl) frameEl.textContent = "frame " + (so.frame_id || "?");
    const srcEl = $("scene-objects-source");
    if (srcEl) srcEl.textContent = "source " + (so.source_node || "?");
  }

  // ADR-0038 — overlay remembered objects on the SLAM 2D map as labelled dots.
  // Reuses the SLAM card's worldToPixel transform (objects are in the same map
  // frame as the robot pose). Appends to the slam-overlay svg AFTER
  // renderRobotMarker so the robot footprint is preserved. Best-effort: a
  // missing map (arm-only deploys) just skips the overlay — the table still shows.
  function renderSceneObjectsOnMap(slam, so) {
    const svg = $("slam-overlay");
    if (!svg) return;
    const objects = (so && Array.isArray(so.objects)) ? so.objects : [];
    const W = slam && slam.width, H = slam && slam.height, res = slam && slam.resolution_m;
    if (!W || !H || !res || objects.length === 0) return;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const ox = slam.origin_x || 0, oy = slam.origin_y || 0;
    const accent = "var(--accent-skill,#ffd166)";
    const r = Math.max(2, 0.12 / res);
    const fontPx = Math.max(6, 0.35 / res);
    let markup = "";
    for (const o of objects) {
      if (o.x == null || o.y == null) continue;
      if (o.frame_id && slam.frame_id && o.frame_id !== slam.frame_id) continue;
      const { px, py } = worldToPixel(o.x, o.y, ox, oy, res, H);
      const text = String(o.label || o.id || "?").replace(/[<>&]/g, "");
      markup +=
        `<circle cx="${px}" cy="${py}" r="${r}" fill="${accent}" fill-opacity="0.9" ` +
        `stroke="#1a1a1a" stroke-width="${Math.max(0.3, r * 0.18)}"/>` +
        `<text x="${px + r * 1.4}" y="${py - r * 0.6}" font-size="${fontPx}" ` +
        `fill="${accent}" stroke="#1a1a1a" stroke-width="${fontPx * 0.04}" ` +
        `paint-order="stroke">${text}</text>`;
    }
    svg.insertAdjacentHTML("beforeend", markup);
  }

  // ADR-0018 F4 — render the latest ReasonerCore tick. Same shape as
  // the SLAM card: empty-state until the first reasoner.tick span
  // lands, then "<tool>" headline + tick_idx / rskill_id / model /
  // force / error_kind metadata pinned below.
  function renderReasoner(r) {
    const ageEl = $("reasoner-age");
    const empty = $("reasoner-empty");
    const detail = $("reasoner-detail");
    if (!r || !r.ts_unix) {
      if (ageEl) ageEl.textContent = "—";
      if (empty) empty.style.display = "block";
      if (detail) detail.style.display = "none";
      return;
    }
    if (ageEl) ageEl.textContent = fmtAge(r.ts_unix);
    if (empty) empty.style.display = "none";
    if (detail) detail.style.display = "block";
    const tool = $("reasoner-tool");
    if (tool) {
      const t = r.tool || (r.suppressed_reason ? "(suppressed: " + r.suppressed_reason + ")" : "(no tool)");
      tool.textContent = t;
    }
    const tick = $("reasoner-tick");
    if (tick) tick.textContent = r.tick_idx != null ? "tick " + r.tick_idx : "";
    const rskill = $("reasoner-rskill");
    if (rskill) rskill.textContent = r.rskill_id ? "rskill " + r.rskill_id : "";
    const model = $("reasoner-model");
    if (model) model.textContent = r.model ? "model " + r.model : "";
    const force = $("reasoner-force");
    if (force) force.textContent = r.force ? "forced" : "";
    const err = $("reasoner-error");
    if (err) err.textContent = r.error_kind ? "error " + r.error_kind : "";
  }

  function gaugeBar(pct) {
    const cls = pct >= 90 ? "crit" : pct >= 75 ? "warn" : "";
    return `<div class="bar"><div class="fill ${cls}" style="width: ${Math.max(0, Math.min(100, pct))}%"></div></div>`;
  }

  function valWithUnits(magnitude, units) {
    return `${magnitude}<span class="sub">${units}</span>`;
  }

  function renderSystem(sys) {
    $("sys-age").textContent = fmtAge(sys && sys.ts_unix);
    const el = $("gauges");
    el.innerHTML = "";
    if (!sys || sys.ts_unix == null) {
      el.innerHTML = '<div class="empty-state">waiting for system metrics</div>';
      return;
    }
    const rows = [];
    if (sys.cpu_util_pct != null) {
      rows.push(["cpu", sys.cpu_util_pct, valWithUnits(num(sys.cpu_util_pct, 0), "%")]);
    }
    if (sys.ram_used_mb != null && sys.ram_total_mb) {
      const pct = (sys.ram_used_mb / sys.ram_total_mb) * 100;
      rows.push([
        "ram",
        pct,
        valWithUnits(num(sys.ram_used_mb / 1024, 1), " / " + num(sys.ram_total_mb / 1024, 1) + " GiB"),
      ]);
    }
    if (sys.gpus) {
      for (const [idx, g] of Object.entries(sys.gpus)) {
        if (g.util_pct != null) rows.push([`gpu${idx}`, g.util_pct, valWithUnits(num(g.util_pct, 0), "%")]);
        if (g.memory_used_mb != null && g.memory_total_mb) {
          const pct = (g.memory_used_mb / g.memory_total_mb) * 100;
          rows.push([
            `gpu${idx} mem`,
            pct,
            valWithUnits(num(g.memory_used_mb / 1024, 1), " / " + num(g.memory_total_mb / 1024, 1) + " GiB"),
          ]);
        }
      }
    }
    if (rows.length === 0) { el.innerHTML = '<div class="empty-state">no system metrics yet</div>'; return; }
    for (const [lbl, pct, val] of rows) {
      const row = document.createElement("div"); row.className = "gauge-row";
      row.innerHTML = `<span class="lbl">${lbl}</span>${gaugeBar(pct)}<span class="v">${val}</span>`;
      el.appendChild(row);
    }
  }

  function renderLedger(safety) {
    $("ledger-age").textContent = fmtAge(safety && safety.latest_ts_unix);
    const el = $("ledger");
    el.innerHTML = "";
    const checks = safety && safety.checks;
    if (!checks || Object.keys(checks).length === 0) {
      el.innerHTML = '<div class="ledger-empty" style="grid-column: 1 / -1">no safety checks recorded yet</div>';
      return;
    }
    const sorted = Object.entries(checks).sort((a, b) => b[1].ts_unix - a[1].ts_unix);
    for (const [name, c] of sorted) {
      const sev = String(c.severity || "info").toLowerCase();
      const kEl = document.createElement("span"); kEl.className = "k"; kEl.textContent = name;
      const vEl = document.createElement("span"); vEl.className = "pill " + sev; vEl.textContent = sev;
      el.appendChild(kEl); el.appendChild(vEl);
    }
  }

  function renderCounters(counters) {
    const map = {
      "cnt-safety": "openral.event.safety_violation",
      "cnt-estop": "openral.event.estop_requested",
      "cnt-deadline": "openral.event.deadline_missed",
      "cnt-sensor": "openral.event.sensor_stale",
    };
    const containerMap = {
      "cnt-safety": "counter-safety",
      "cnt-estop": "counter-estop",
      "cnt-deadline": "counter-deadline",
      "cnt-sensor": "counter-sensor",
    };
    for (const [elId, key] of Object.entries(map)) {
      const v = counters[key] || 0;
      $(elId).textContent = v;
      const cardEl = $(containerMap[elId]);
      if (cardEl) cardEl.classList.toggle("zero", v === 0);
    }
  }

  function metaFromAttrs(ev) {
    if (!ev.attrs) return "";
    const dur = ev.attrs["duration_ms"] ?? ev.attrs["openral.duration_ms"];
    if (dur != null && !isNaN(Number(dur))) return Number(dur).toFixed(1) + " ms";
    const tick = ev.attrs["openral.tick.idx"] ?? ev.attrs["tick.idx"];
    if (tick != null) return "tick " + tick;
    return "";
  }

  // Event-log severity filter. Four buckets: debug / info / warn / error.
  // `error` catches safety_violation, estop_requested, error_latched, fatal
  // log lines + anything unrecognised. `debug` carries the bridged structlog
  // DEBUG lines (issue #318) and defaults OFF — high-rate DEBUG (world_state
  // ~30 Hz) would otherwise flood the 60-event view; toggle it on when needed.
  const eventSevFilter = { debug: false, info: true, warn: true, error: true };
  const sevBucket = (sev) =>
    (sev === "debug" || sev === "info" || sev === "warn") ? sev : "error";
  let _lastEvents = [];

  function renderEvents(events) {
    _lastEvents = events || [];
    const el = $("events");
    const all = _lastEvents;
    // Per-bucket counts for the chip badges (over the full, unfiltered set).
    const counts = { debug: 0, info: 0, warn: 0, error: 0 };
    for (const ev of all) counts[sevBucket(String(ev.severity || "info").toLowerCase())]++;
    for (const chip of document.querySelectorAll("#event-filters .filter-chip")) {
      const b = chip.dataset.sev;
      const cnt = chip.querySelector(".cnt");
      if (cnt) cnt.textContent = counts[b];
    }
    const shown = all.filter((ev) => eventSevFilter[sevBucket(String(ev.severity || "info").toLowerCase())]);
    if (all.length === 0) {
      el.innerHTML = '<div class="empty-state">No events yet.</div>';
      return;
    }
    if (shown.length === 0) {
      el.innerHTML = '<div class="empty-state">No events match the active filters.</div>';
      return;
    }
    el.innerHTML = "";
    for (const ev of shown.slice(0, 60)) {
      const sev = String(ev.severity || "info").toLowerCase();
      const row = document.createElement("div");
      row.className = "event " + sev;
      const body = ev.title || ev.kind || "";
      const meta = metaFromAttrs(ev);
      row.innerHTML = `
        <span class="ts">${fmtTime(ev.ts_unix)}</span>
        <span class="lvl">${sev}</span>
        <span class="body">${body}</span>
        <span class="meta">${meta}</span>
      `;
      el.appendChild(row);
    }
  }

  // Strip noisy top-level prefixes so metric names fit in the card without
  // losing meaningful context. The filter chips already label the namespace
  // (world_state / system / sdk / …), so repeating it in every row is noise.
  const _METRIC_PREFIXES = [
    "openral.world_state.",
    "openral.system.",
    "otel.sdk.",
    "openral.",
  ];
  function stripMetricPrefix(n) {
    for (const p of _METRIC_PREFIXES) {
      if (n.startsWith(p)) return n.slice(p.length);
    }
    return n;
  }

  function nameMarkup(n) {
    const stripped = stripMetricPrefix(n);
    const idx = stripped.lastIndexOf(".");
    if (idx < 0) return stripped;
    const ns = stripped.slice(0, idx + 1);
    const leaf = stripped.slice(idx + 1);
    const nsSpan = document.createElement("span");
    nsSpan.className = "ns";
    nsSpan.textContent = ns;
    const frag = document.createDocumentFragment();
    frag.appendChild(nsSpan);
    frag.appendChild(document.createTextNode(leaf));
    return frag;
  }

  function sparkline(svgEl, samples) {
    if (!samples || samples.length < 2) return;
    const w = 240, h = 26;
    const values = samples.map((s) => s[1]);
    const min = Math.min(...values), max = Math.max(...values);
    const span = (max - min) || 1;
    const pts = samples.map((s, i) => {
      const x = (i / (samples.length - 1)) * w;
      const y = h - 2 - ((s[1] - min) / span) * (h - 4);
      return x.toFixed(1) + "," + y.toFixed(1);
    }).join(" ");
    svgEl.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svgEl.setAttribute("preserveAspectRatio", "none");
    svgEl.innerHTML = `<polyline points="${pts}"/>`;
  }

  // Metrics grouping + group filter (issue 2/13). Filter buttons select ALL or
  // a single semantic namespace (system / world_state / sdk / hal / …) derived
  // from the metric name; when ALL is active rows are grouped under namespace
  // headers. Chips are rendered dynamically from whatever namespaces are
  // present, so new subsystems appear automatically.
  let metricGroup = "all";
  let _lastMetrics = [];

  // Namespace = the subsystem segment of the dotted metric name. The OpenRAL
  // SDK self-metrics (process/runtime/exporter telemetry) collapse under "sdk".
  function metricNamespace(name) {
    const parts = String(name || "").split(".");
    if (parts[0] === "openral" && parts.length > 1) {
      const seg = parts[1];
      // system / world_state / hal / rskill / reasoner / safety stay as-is;
      // everything else openral.* is SDK-level instrumentation.
      const known = ["system", "world_state", "hal", "rskill", "reasoner", "safety", "perception"];
      return known.includes(seg) ? seg : "sdk";
    }
    return parts[0] || "other";
  }

  // Render "k=v" label suffix so per-component series (e.g. four
  // world_state.staleness_ms keyed by component=) are distinguishable rather
  // than looking like duplicates.
  function labelSuffix(labels) {
    if (!labels) return "";
    const entries = Object.entries(labels)
      .map(([k, v]) => [k.split(".").slice(-1)[0], v])
      .filter(([k]) => k !== "gpu" );  // gpu index already shown in System card
    if (!entries.length) return "";
    return " {" + entries.map(([k, v]) => `${k}=${v}`).join(", ") + "}";
  }

  function metricRow(m) {
    const row = document.createElement("div"); row.className = "metric-row";
    const name = document.createElement("span"); name.className = "name";
    name.appendChild(nameMarkup(m.name));
    const suffix = labelSuffix(m.labels);
    if (suffix) {
      const sfx = document.createElement("span");
      sfx.className = "ns"; sfx.textContent = suffix;
      name.appendChild(sfx);
    }
    const unit = document.createElement("span"); unit.className = "unit"; unit.textContent = m.unit || m.kind;
    const p50 = document.createElement("span"); p50.className = "pct";
    const p95 = document.createElement("span"); p95.className = "pct";
    const latest = document.createElement("span"); latest.className = "latest";
    if (m.kind === "histogram") {
      p50.innerHTML = m.p50 != null ? `<span class="lbl">p50</span>${m.p50.toFixed(1)}` : "";
      p95.innerHTML = m.p95 != null ? `<span class="lbl">p95</span>${m.p95.toFixed(1)}` : "";
      latest.innerHTML = `<span class="n">n=${m.samples ? m.samples.length : 0}</span>`;
    } else if (m.kind === "sum") {
      latest.textContent = (m.cumulative != null ? Number(m.cumulative).toFixed(1) : "0");
    } else {
      latest.textContent = m.latest != null ? Number(m.latest).toFixed(2) : "—";
    }
    const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
    svg.classList.add("spark");
    sparkline(svg, m.samples);
    row.appendChild(name); row.appendChild(unit); row.appendChild(p50); row.appendChild(p95); row.appendChild(latest); row.appendChild(svg);
    return row;
  }

  let _metricChipSig = "";  // signature of the current chip set (namespace order)
  function renderMetricChips(counts) {
    const bar = $("metric-filters");
    if (!bar) return;
    const namespaces = Object.keys(counts).filter((k) => k !== "all").sort();
    const order = ["all", ...namespaces];
    const sig = order.join("|");
    // Rebuild chips ONLY when the namespace set changes. Re-creating them on
    // every SSE tick (~1/s) made clicks unreliable — a mousedown could land on
    // a chip that got replaced before mouseup, so the click never fired. Stable
    // chips + in-place updates fix the "filter buttons don't work" bug.
    if (sig !== _metricChipSig) {
      _metricChipSig = sig;
      for (const c of bar.querySelectorAll(".filter-chip")) c.remove();
      for (const ns of order) {
        const chip = document.createElement("span");
        chip.className = "filter-chip";
        chip.dataset.mgroup = ns;
        const lbl = document.createElement("span"); lbl.textContent = ns;
        const cnt = document.createElement("span"); cnt.className = "cnt";
        chip.appendChild(lbl); chip.appendChild(cnt);
        chip.addEventListener("click", () => {
          metricGroup = ns;
          // Reflect the selection immediately (don't wait for the next tick).
          for (const c of bar.querySelectorAll(".filter-chip")) {
            c.classList.toggle("active", c.dataset.mgroup === metricGroup);
          }
          renderMetrics(_lastMetrics);
        });
        bar.appendChild(chip);
      }
    }
    // Update counts + active state in place every tick (no DOM churn).
    for (const chip of bar.querySelectorAll(".filter-chip")) {
      const ns = chip.dataset.mgroup;
      chip.classList.toggle("active", ns === metricGroup);
      const cnt = chip.querySelector(".cnt");
      if (cnt) cnt.textContent = counts[ns] ?? 0;
    }
  }

  function renderMetrics(metrics) {
    _lastMetrics = metrics || [];
    const el = $("metrics");
    // Per-namespace counts for the chip badges (over the full, unfiltered set).
    const counts = { all: _lastMetrics.length };
    for (const m of _lastMetrics) {
      const ns = metricNamespace(m.name);
      counts[ns] = (counts[ns] || 0) + 1;
    }
    // If the active group no longer exists (metrics changed), fall back to ALL.
    if (metricGroup !== "all" && !counts[metricGroup]) metricGroup = "all";
    renderMetricChips(counts);
    if (_lastMetrics.length === 0) {
      el.innerHTML = '<div class="empty-state">No metrics ingested yet — point a workload at this port with <code>OTEL_EXPORTER_OTLP_ENDPOINT</code>.</div>';
      return;
    }
    const shown = metricGroup === "all"
      ? _lastMetrics
      : _lastMetrics.filter((m) => metricNamespace(m.name) === metricGroup);
    if (shown.length === 0) {
      el.innerHTML = '<div class="empty-state">No ' + metricGroup + ' metrics ingested yet.</div>';
      return;
    }
    el.innerHTML = "";
    const sorted = shown.slice().sort(
      (a, b) => metricNamespace(a.name).localeCompare(metricNamespace(b.name)) || a.name.localeCompare(b.name)
    );
    let lastNs = null;
    for (const m of sorted) {
      const ns = metricNamespace(m.name);
      // Group headers only when showing ALL (a single-namespace view needs none).
      if (metricGroup === "all" && ns !== lastNs) {
        lastNs = ns;
        const hd = document.createElement("div");
        hd.className = "metric-group-hd";
        hd.textContent = ns + " · " + counts[ns];
        el.appendChild(hd);
      }
      el.appendChild(metricRow(m));
    }
  }

  // Jaeger UI url is operator-configured via the OPENRAL_JAEGER_UI_URL
  // env var on the dashboard process, surfaced through /api/config.
  // When unset (the common case — no Jaeger running) we keep the link
  // disabled with a tooltip explaining how to enable it, instead of
  // pointing at a guessed `localhost:16686` that produces a broken-link
  // click for everyone who doesn't run Jaeger locally.
  let JAEGER_URL = "";
  fetch("/api/config")
    .then((r) => r.ok ? r.json() : {})
    .then((cfg) => { JAEGER_URL = (cfg && cfg.jaeger_ui_url) ? String(cfg.jaeger_ui_url).replace(/\/$/, "") : ""; })
    .catch(() => { JAEGER_URL = ""; });

  function renderTrace(trace) {
    const el = $("trace-id");
    const linkEl = $("jaeger-link");
    const tid = trace && trace.latest_trace_id;
    if (!tid) {
      el.textContent = "—";
      linkEl.classList.remove("active");
      linkEl.classList.add("disabled");
      linkEl.removeAttribute("href");
      linkEl.title = "no traces ingested yet";
      return;
    }
    el.textContent = tid.slice(0, 16) + "…";
    el.title = tid + " (click to copy)";
    el.onclick = () => {
      navigator.clipboard?.writeText(tid);
      el.classList.add("copied");
      setTimeout(() => el.classList.remove("copied"), 800);
    };
    if (!JAEGER_URL) {
      linkEl.classList.remove("active");
      linkEl.classList.add("disabled");
      linkEl.removeAttribute("href");
      linkEl.title =
        "set OPENRAL_JAEGER_UI_URL on the `openral dashboard` process to a reachable " +
        "Jaeger UI (e.g. http://localhost:16686) to enable this link";
      return;
    }
    linkEl.classList.add("active");
    linkEl.classList.remove("disabled");
    linkEl.href = `${JAEGER_URL}/trace/${tid}`;
    linkEl.title = `open trace ${tid} in ${JAEGER_URL}`;
  }

  // Pulse-on-update tracking: remember which cards' timestamps changed
  // since the last render and briefly highlight their borders.
  const _lastSeen = new Map();
  function pulseIfNew(elId, ts) {
    if (!ts) return;
    const prev = _lastSeen.get(elId);
    if (prev !== ts) {
      _lastSeen.set(elId, ts);
      const el = document.getElementById(elId);
      if (!el || prev === undefined) return;
      el.classList.add("pulse");
      setTimeout(() => el.classList.remove("pulse"), 700);
    }
  }

  // Per-card status dot (next to the title). Four states, derived from data
  // freshness so every card reads at a glance like the header conn dot:
  //   wait   = blinking white — no data has ever arrived
  //   live   = green          — data received within STALE_S
  //   paused = yellow         — was receiving, but latest data is stale
  //   error  = red            — the card reported an error span
  const STALE_S = 10;
  const _dotSeen = new Set();
  const DOT_LABEL = {
    wait: "waiting for data", live: "receiving data",
    paused: "data paused (stale)", error: "error",
  };
  function setDot(cardId, ts, isError) {
    const el = document.getElementById(cardId);
    if (!el) return;
    const title = el.querySelector(".title");
    if (!title) return;
    let st;
    if (isError) st = "error";
    else if (ts) { _dotSeen.add(cardId); st = (Date.now() / 1000 - ts) < STALE_S ? "live" : "paused"; }
    else st = _dotSeen.has(cardId) ? "paused" : "wait";
    title.classList.remove("st-wait", "st-live", "st-paused", "st-error");
    title.classList.add("st-" + st);
    title.title = DOT_LABEL[st];  // a11y: state is not conveyed by colour alone
  }

  function render(state) {
    renderIdentity(state);
    const cards = state.cards || {};
    const liveSkill = cards.rskill_execute || cards.rskill_tick || cards.rskill_activate || null;
    renderCard("rskill_execute", liveSkill);
    renderCard("inference", cards.inference || null);
    renderCard("safety", cards.safety || null);
    if (liveSkill) pulseIfNew("card-rskill_execute", liveSkill.ts_unix);
    if (cards.inference) pulseIfNew("card-inference", cards.inference.ts_unix);
    if (cards.safety) pulseIfNew("card-safety", cards.safety.ts_unix);

    const topics = state.topics || {};
    renderRobotState(topics.robot_state, topics.commands);
    renderCommands(topics.commands);
    renderWorldState(topics.world_state);
    renderPerception(topics.perception);
    renderSlamMap(topics.slam);
    renderSceneObjectsOnMap(topics.slam, topics.scene_objects);
    renderWorldCloud(topics.pointcloud);
    renderSceneObjects(topics.scene_objects);
    renderReasoner(topics.reasoner);
    renderSystem(topics.system);
    renderLedger(topics.safety);
    renderTrace(topics.trace);

    pulseIfNew("card-robot-state", topics.robot_state && topics.robot_state.ts_unix);
    pulseIfNew("card-commands", topics.commands && topics.commands.ts_unix);
    pulseIfNew("card-world-state", topics.world_state && topics.world_state.ts_unix);
    pulseIfNew("card-system", topics.system && topics.system.ts_unix);
    pulseIfNew("card-safety-ledger", topics.safety && topics.safety.latest_ts_unix);
    pulseIfNew("card-slam-map", topics.slam && topics.slam.ts_unix);
    pulseIfNew("card-world-cloud", topics.pointcloud && topics.pointcloud.ts_unix);
    pulseIfNew("card-scene-objects", topics.scene_objects && topics.scene_objects.ts_unix);
    pulseIfNew("card-reasoner", topics.reasoner && topics.reasoner.ts_unix);

    // Status dot per card — same 4-state logic as the header conn dot.
    setDot("card-rskill_execute", liveSkill && liveSkill.ts_unix, !!(liveSkill && liveSkill.status_code === 2));
    setDot("card-inference", cards.inference && cards.inference.ts_unix, !!(cards.inference && cards.inference.status_code === 2));
    setDot("card-safety", cards.safety && cards.safety.ts_unix, !!(cards.safety && cards.safety.status_code === 2));
    setDot("card-robot-state", topics.robot_state && topics.robot_state.ts_unix, false);
    setDot("card-commands", topics.commands && topics.commands.ts_unix, false);
    setDot("card-world-state", topics.world_state && topics.world_state.ts_unix, false);
    setDot("card-system", topics.system && topics.system.ts_unix, false);
    setDot("card-safety-ledger", topics.safety && topics.safety.latest_ts_unix, false);
    setDot("card-slam-map", topics.slam && topics.slam.ts_unix, false);
    setDot("card-world-cloud", topics.pointcloud && topics.pointcloud.ts_unix, false);
    setDot("card-scene-objects", topics.scene_objects && topics.scene_objects.ts_unix, false);
    setDot("card-reasoner", topics.reasoner && topics.reasoner.ts_unix, false);

    renderCounters(state.counters || {});
    renderEvents(state.events || []);
    renderMetrics(state.metrics || []);

    const conn = $("conn");
    const label = $("conn-label");
    if (!state.last_ingest_ts) {
      conn.className = "conn wait"; label.textContent = "waiting…";
    } else {
      const dt = Date.now() / 1000 - state.last_ingest_ts;
      if (dt < 10) { conn.className = "conn live"; label.textContent = "live"; }
      else if (dt < 60) { conn.className = "conn stale"; label.textContent = "stale (" + dt.toFixed(0) + "s)"; }
      else { conn.className = "conn dead"; label.textContent = "dead (" + Math.floor(dt / 60) + "m)"; }
    }
  }

  let es = null;
  function connect() {
    if (es) es.close();
    es = new EventSource("/api/stream");
    es.onmessage = (m) => {
      try { render(JSON.parse(m.data)); } catch (e) { /* malformed: ignore */ }
    };
    es.onerror = () => {
      $("conn").className = "conn stale"; $("conn-label").textContent = "reconnecting…";
      es.close();
      setTimeout(connect, 1500);
    };
  }
  connect();
  setInterval(() => {
    fetch("/api/state").then((r) => r.json()).then(render).catch(() => {});
  }, 5000);

  // ── Operator prompt (POSTs to /api/prompt → `openral prompt` → prompt_router) ──
  const promptInput = $("prompt-input");
  const promptSend = $("prompt-send");
  const promptStatus = $("prompt-status");
  function setPromptStatus(text, kind) {
    promptStatus.textContent = text;
    promptStatus.className = "status" + (kind ? " " + kind : "");
  }
  // The voice prompt "arms" a brief auto-send countdown after transcription so a
  // mis-recognition can be caught before it reaches the robot. Any explicit send
  // (Send / Enter), a cancel (Esc / editing the text), or starting a new
  // recording clears it so it can never double-fire.
  let autoSendTimer = null;
  let autoSendTick = null;
  function clearAutoSend() {
    if (autoSendTimer) { clearTimeout(autoSendTimer); autoSendTimer = null; }
    if (autoSendTick) { clearInterval(autoSendTick); autoSendTick = null; }
  }
  async function sendPrompt() {
    clearAutoSend();
    const text = promptInput.value.trim();
    if (!text) return;
    promptSend.disabled = true;
    setPromptStatus("publishing…", "");
    try {
      const resp = await fetch("/api/prompt", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: text }),
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) {
        setPromptStatus(body.error || ("HTTP " + resp.status), "err");
      } else {
        setPromptStatus("published", "ok");
        promptInput.value = "";
        setTimeout(() => setPromptStatus("", ""), 3000);
      }
    } catch (e) {
      setPromptStatus(String(e), "err");
    } finally {
      promptSend.disabled = false;
    }
  }
  promptSend.addEventListener("click", sendPrompt);
  promptInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendPrompt(); }
    else if (e.key === "Escape" && autoSendTimer) {
      clearAutoSend();
      setPromptStatus("auto-send cancelled — edit, then press Send", "");
    }
  });
  // Typing into the transcript cancels the pending auto-send (you're correcting
  // it). Programmatic fills set `.value` directly, which does not fire "input".
  promptInput.addEventListener("input", () => {
    if (autoSendTimer) {
      clearAutoSend();
      setPromptStatus("auto-send cancelled — press Send when ready", "");
    }
  });

  // ── Voice prompt (mic → local STT) ──────────────────────────────────────────
  // Click the mic to start: we lazy-load ricky0123/vad-web (Silero VAD) on first
  // use and listen. Two ways to stop, both transcribe: the VAD auto-detects when
  // you stop speaking (onSpeechEnd), OR you click the mic again to stop now (we
  // accumulate frames via onFrameProcessed so a manual stop has audio to send).
  // We encode the captured 16 kHz samples to a mono WAV and POST it to
  // /api/transcribe, which runs a LOCAL faster-whisper model on the host. The
  // returned text fills the prompt box and is sent via the normal sendPrompt().
  // Fully offline: nothing leaves the machine and nothing is fetched from a CDN.
  // The VAD library, Silero model, onnxruntime-web and its wasm are all vendored
  // under /static/vendor/vad/ (see that directory's NOTICE.md for versions).
  const VAD_VENDOR = "/static/vendor/vad/";
  const VAD_WEB_SRC = `${VAD_VENDOR}bundle.min.js`;
  const ORT_SRC = `${VAD_VENDOR}ort.wasm.min.js`;
  const VAD_ASSET_BASE = VAD_VENDOR;   // worklet + silero_vad_*.onnx
  const VAD_WASM_BASE = VAD_VENDOR;    // ort-wasm-simd-threaded.{wasm,mjs}

  const promptMic = $("prompt-mic");
  const micMeter = $("mic-meter");
  const meterBars = micMeter ? Array.from(micMeter.querySelectorAll("i")) : [];
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  let micVad = null;          // MicVAD instance, created once on first use.
  let micListening = false;
  let capturedFrames = [];   // every 16 kHz frame while listening — lets a manual stop transcribe.
  let finalizing = false;    // guard so the auto- and manual-stop paths can't double-fire.
  // Reliable auto-stop: rather than depend solely on the VAD's own onSpeechEnd
  // (which can be slow or never fire in a noisy room), we watch the model's
  // per-frame speech probability ourselves and finalize after a fixed silence
  // window once speech has been seen.
  const VOICE_PROB = 0.5;     // per-frame isSpeech above this = voice present
  const SILENCE_MS = 1100;    // finalize after this much silence following speech
  let speechSeen = false;
  let lastVoiceAt = 0;

  function setMicState(state) {  // "idle" | "listening" | "working"
    if (!promptMic) return;
    promptMic.dataset.state = state;
    promptMic.setAttribute("aria-pressed", state === "listening" ? "true" : "false");
    promptMic.title =
      state === "listening" ? "Listening — click to stop"
      : state === "working" ? "Transcribing…"
      : "Speak a prompt";
  }

  function loadScript(src) {
    return new Promise((resolve, reject) => {
      if (document.querySelector(`script[src="${src}"]`)) { resolve(); return; }
      const s = document.createElement("script");
      s.src = src;
      s.onload = () => resolve();
      s.onerror = () => reject(new Error("failed to load " + src));
      document.head.appendChild(s);
    });
  }

  // Float32 PCM [-1,1] @ sampleRate (mono) → 16-bit WAV Blob — the container
  // faster-whisper/PyAV ingests directly. Avoids depending on a vad-web util.
  function encodeWav(samples, sampleRate) {
    const n = samples.length;
    const buf = new ArrayBuffer(44 + n * 2);
    const view = new DataView(buf);
    const str = (off, s) => { for (let i = 0; i < s.length; i++) view.setUint8(off + i, s.charCodeAt(i)); };
    str(0, "RIFF"); view.setUint32(4, 36 + n * 2, true); str(8, "WAVE");
    str(12, "fmt "); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
    view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
    str(36, "data"); view.setUint32(40, n * 2, true);
    let off = 44;
    for (let i = 0; i < n; i++) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      off += 2;
    }
    return new Blob([view], { type: "audio/wav" });
  }

  // Join the accumulated per-frame Float32Arrays into one contiguous buffer.
  function concatFrames(frames) {
    let len = 0;
    for (const f of frames) len += f.length;
    const out = new Float32Array(len);
    let off = 0;
    for (const f of frames) { out.set(f, off); off += f.length; }
    return out;
  }

  // ── Live level meter: drive the equalizer bars from the mic's RMS so the
  // operator can see the dashboard is hearing them. Animated only via
  // transform: scaleY (no layout reflow); skipped under reduced-motion.
  function frameRms(frame) {
    let sum = 0;
    for (let i = 0; i < frame.length; i++) sum += frame[i] * frame[i];
    return Math.sqrt(sum / frame.length);
  }
  function resetMeter() {
    for (const bar of meterBars) bar.style.transform = "scaleY(0.18)";
  }
  function updateMeter(rms) {
    if (!meterBars.length || reduceMotion) return;
    const level = Math.min(1, rms * 7);   // map typical speech RMS (~0–0.15) to 0–1
    for (let i = 0; i < meterBars.length; i++) {
      // taller in the centre for a natural equalizer shape
      const weight = 0.55 + 0.45 * Math.sin(((i + 1) / (meterBars.length + 1)) * Math.PI);
      const s = 0.18 + level * weight * 0.82;
      meterBars[i].style.transform = `scaleY(${Math.min(1, s).toFixed(3)})`;
    }
  }
  function showMeter(on) {
    if (!micMeter) return;
    micMeter.classList.toggle("active", on);
    if (on) resetMeter();
  }

  // Arm the cancellable auto-send countdown after a transcription fills the box.
  function armAutoSend() {
    clearAutoSend();
    const DELAY_MS = 1500;
    let remaining = DELAY_MS;
    const render = () => setPromptStatus("sending in " + (remaining / 1000).toFixed(1) + "s — Esc to cancel", "");
    render();
    autoSendTick = setInterval(() => { remaining -= 100; if (remaining > 0) render(); }, 100);
    autoSendTimer = setTimeout(() => { clearAutoSend(); sendPrompt(); }, DELAY_MS);
  }

  async function transcribeAndSend(samples) {
    setMicState("working");
    setPromptStatus("transcribing…", "");
    try {
      const resp = await fetch("/api/transcribe", {
        method: "POST",
        headers: { "Content-Type": "audio/wav" },
        body: encodeWav(samples, 16000),
      });
      const body = await resp.json().catch(() => ({}));
      if (!resp.ok) { setPromptStatus(body.error || ("HTTP " + resp.status), "err"); return; }
      const text = (body.text || "").trim();
      if (!text) { setPromptStatus("no speech recognized", "err"); return; }
      promptInput.value = text;
      promptInput.focus();
      armAutoSend();               // fill the box, then auto-send unless cancelled.
    } catch (e) {
      setPromptStatus(String(e), "err");
    } finally {
      capturedFrames = [];
      finalizing = false;
      setMicState("idle");
    }
  }

  // Single exit for both paths — silence detection (auto) and the operator
  // clicking the mic to stop (manual). Pauses the VAD and transcribes
  // `samples`; the guard keeps the two paths from racing into a double-send.
  function finalizeCapture(samples) {
    if (finalizing) return;
    finalizing = true;
    micListening = false;
    if (micVad) micVad.pause();
    showMeter(false);
    if (!samples || samples.length === 0) {
      capturedFrames = [];
      finalizing = false;
      setMicState("idle");
      setPromptStatus("no speech recorded", "err");
      return;
    }
    transcribeAndSend(samples);
  }

  async function ensureVad() {
    if (micVad) return micVad;
    setPromptStatus("loading speech model…", "");
    await loadScript(ORT_SRC);
    await loadScript(VAD_WEB_SRC);
    micVad = await window.vad.MicVAD.new({
      baseAssetPath: VAD_ASSET_BASE,
      onnxWASMBasePath: VAD_WASM_BASE,
      onFrameProcessed: (probs, frame) => {
        if (!micListening || finalizing || !frame) return;
        capturedFrames.push(frame.slice());   // so a manual stop has audio to send
        updateMeter(frameRms(frame));          // live level meter
        // Our own end-of-speech detector: once speech has been seen, finalize
        // after SILENCE_MS of sub-threshold frames. Robust where the VAD's own
        // onSpeechEnd is slow or never fires.
        const now = Date.now();
        if (probs && probs.isSpeech > VOICE_PROB) { speechSeen = true; lastVoiceAt = now; }
        else if (speechSeen && now - lastVoiceAt > SILENCE_MS) {
          finalizeCapture(concatFrames(capturedFrames));
        }
      },
      onSpeechStart: () => { speechSeen = true; lastVoiceAt = Date.now(); setPromptStatus("listening…", ""); },
      onSpeechEnd: (samples) => finalizeCapture(samples),  // VAD's own trimmed segment (whichever fires first)
    });
    return micVad;
  }

  async function startListening() {
    try {
      const v = await ensureVad();
      clearAutoSend();
      capturedFrames = [];
      finalizing = false;
      speechSeen = false;
      lastVoiceAt = Date.now();
      v.start();
      micListening = true;
      showMeter(true);
      setMicState("listening");
      setPromptStatus("listening… speak, then pause (or click the mic)", "");
    } catch (e) {
      setPromptStatus("mic unavailable: " + (e && e.message ? e.message : e), "err");
      setMicState("idle");
    }
  }

  // Operator clicked the mic to stop — finalize with everything captured so
  // far, instead of waiting on (or depending on) the VAD's silence detector.
  function manualStop() {
    finalizeCapture(concatFrames(capturedFrames));
  }

  if (promptMic) {
    promptMic.addEventListener("click", () => {
      if (micListening) manualStop(); else startListening();
    });
  }

  // ── E-stop recovery (POST /api/estop_reset → ros2 service call) ──
  // A latched safety e-stop makes the kernel drop every command, so no prompt
  // works until it's cleared. This button calls the kernel reset service; on
  // success the operator can send a fresh prompt to resume.
  const estopReset = $("estop-reset");
  if (estopReset) {
    estopReset.addEventListener("click", async () => {
      estopReset.disabled = true;
      setPromptStatus("resetting e-stop…", "");
      try {
        const resp = await fetch("/api/estop_reset", { method: "POST" });
        const body = await resp.json().catch(() => ({}));
        if (resp.ok && body.accepted) {
          setPromptStatus("e-stop cleared — send a prompt to resume", "ok");
        } else if (resp.status === 409) {
          setPromptStatus("reset rejected (cooldown) — wait a moment and retry", "err");
        } else {
          setPromptStatus(body.error || ("HTTP " + resp.status), "err");
        }
      } catch (e) {
        setPromptStatus(String(e), "err");
      } finally {
        estopReset.disabled = false;
      }
    });
  }

  // ── Event-log severity filter (issue 12) — toggle buckets, re-render cache ──
  for (const chip of document.querySelectorAll("#event-filters .filter-chip")) {
    chip.addEventListener("click", () => {
      const b = chip.dataset.sev;
      eventSevFilter[b] = !eventSevFilter[b];
      chip.classList.toggle("active", eventSevFilter[b]);
      renderEvents(_lastEvents);
    });
  }

  // Metrics group filter chips (issue 2) are rendered + wired dynamically in
  // renderMetricChips() since the namespace set depends on live data.

})();
