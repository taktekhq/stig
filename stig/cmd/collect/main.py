"""
Command collect records training laps for Horizon Drive using an oracle
controller (clean, perturbed recovery, or DAgger on-policy labeling).

Outputs standard frame_<unix_ns>_<throttle>_<steering>.jpeg files into
RECORDINGS_DIR, compatible with `stig.cmd.train.main`.
"""

import argparse
import base64
import json
import os
import random
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

GAME_URL = "https://www.shopify.com/ca/editions/summer2025/drive"

ORACLE_JS = """() => {
  const s = window.__stig;
  if (!s) return null;
  const k = s.K.getState();
  const q = s.Q.getState();
  const track = k.track;
  if (!track || !track.segments || track.segments.length === 0) return {ready: false};
  const segs = track.segments;
  const idx = Math.max(0, Math.min(segs.length - 1, q.segmentIndex || 0));
  const seg = segs[idx];

  const dx = q.position.x - seg.position.x;
  const dy = q.position.y - seg.position.y;
  const dz = q.position.z - seg.position.z;
  const lat = dx * seg.binormal.x + dy * seg.binormal.y + dz * seg.binormal.z;

  const fwd = q.forward;
  const vel = q.velocity;
  const speedMs = Math.hypot(vel.x, vel.y, vel.z);

  // Upcoming curvature over next 65m
  let maxCurv = 0;
  let cIdx = idx;
  const t0 = seg.tangent;
  while (cIdx < segs.length - 1 && segs[cIdx].cumulativeLength < seg.cumulativeLength + 65.0) {
    const t1 = segs[cIdx].tangent;
    const dot = Math.max(-1, Math.min(1, t0.x*t1.x + t0.y*t1.y + t0.z*t1.z));
    const ang = Math.acos(dot);
    if (ang > maxCurv) maxCurv = ang;
    cIdx += 2;
  }

  // Adaptive lookahead: phase-lead anticipates turns cleanly at high speed, tighter when recovering from walls
  let lookDist = Math.max(16.0, Math.min(52.0, speedMs * 0.68 + 10.0));
  if (Math.abs(lat) > 4.5 || q.collisionState !== -1) {
    lookDist = Math.min(lookDist, Math.max(13.0, speedMs * 0.42 + 8.0));
  }
  const targetCum = seg.cumulativeLength + lookDist;
  let laIdx = idx;
  while (laIdx < segs.length - 1 && segs[laIdx].cumulativeLength < targetCum) laIdx++;
  const laSeg = segs[laIdx];

  // Car local right vector = forward x up
  const rx = fwd.y * q.up.z - fwd.z * q.up.y;
  const ry = fwd.z * q.up.x - fwd.x * q.up.z;
  const rz = fwd.x * q.up.y - fwd.y * q.up.x;

  // Target point on centerline (slightly damped away from outer wall if far off-center)
  let targetLat = 0.0;
  if (Math.abs(lat) > 4.0) {
    targetLat = -Math.sign(lat) * 0.8;
  }
  const px = laSeg.position.x + laSeg.binormal.x * targetLat;
  const py = laSeg.position.y + laSeg.binormal.y * targetLat;
  const pz = laSeg.position.z + laSeg.binormal.z * targetLat;

  const tx = px - q.position.x;
  const ty = py - q.position.y;
  const tz = pz - q.position.z;
  const rightProj = tx * rx + ty * ry + tz * rz;
  const fwdProj = tx * fwd.x + ty * fwd.y + tz * fwd.z;
  const velRight = vel.x * rx + vel.y * ry + vel.z * rz;

  const steerAngle = Math.atan2(rightProj - velRight * 0.25, Math.max(1.0, fwdProj));

  return {
    ready: true,
    screen: k.gameScreen,
    raceStartTime: k.raceStartTime,
    raceEndTime: k.raceEndTime,
    progress: q.normalizedProgress,
    speed: q.speed,
    trackWidth: track.width,
    lat: lat,
    steerAngle: steerAngle,
    maxCurv: maxCurv,
    collision: q.collisionState,
    isInJump: q.isInJump,
  };
}"""


def oracle_action(st: dict) -> tuple[str, str]:
    """Compute expert (throttle, steering) action from game state."""
    if st["isInJump"]:
        return ("accelerate", "")

    ang = st["steerAngle"]
    lat = st["lat"]
    col = st["collision"]

    # Strong wall-recovery override when touching a barrier
    if col != -1:
        s_act = "left" if lat > 0 else "right"
    elif ang > 0.04:
        s_act = "right"
    elif ang < -0.04:
        s_act = "left"
    else:
        s_act = ""

    spd = st["speed"]
    curv = st["maxCurv"]
    if spd < 140:
        t_act = "accelerate"
    elif curv > 1.25 and spd > 195:
        t_act = "brake"
    elif curv > 0.95 and spd > 215:
        t_act = ""
    else:
        t_act = "accelerate"

    return (t_act, s_act)


def query_agent(agent_url: str, img_bytes: bytes) -> tuple[str, str]:
    req = urllib.request.Request(
        f"{agent_url.rstrip('/')}/act",
        data=img_bytes,
        headers={"Content-Type": "image/jpeg"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=2.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return (data.get("throttle", ""), data.get("steering", ""))


def collect(
    cdp_url: str,
    recordings_dir: str,
    modes: list[str],
    fps: int = 10,
    agent_url: str = "http://localhost:8080",
    seed: int = 42,
):
    rng = random.Random(seed)
    session = f"session_{time.strftime('%Y%m%dT%H%M%S')}"
    session_dir = Path(recordings_dir) / session
    session_dir.mkdir(parents=True, exist_ok=True)
    interval = 1.0 / fps

    with sync_playwright() as p:
        browser = p.chromium.connect_over_cdp(cdp_url)
        ctx = browser.contexts[0]
        page = ctx.new_page()
        page.set_viewport_size({"width": 960, "height": 600})
        cdp = ctx.new_cdp_session(page)

        def capture_frame() -> bytes:
            res = cdp.send(
                "Page.captureScreenshot",
                {
                    "format": "jpeg",
                    "quality": 90,
                    "clip": {"x": 0, "y": 0, "width": 960, "height": 600, "scale": 1},
                    "fromSurface": False,
                    "captureBeyondViewport": False,
                    "optimizeForSpeed": True,
                },
            )
            return base64.b64decode(res["data"])

        def handle_route(route):
            resp = route.fetch()
            body = resp.text()
            if "export{" in body and "Q as d" in body:
                body = body.replace("export{", "window.__stig={Q,K,Ge,Qe};export{", 1)
            route.fulfill(response=resp, body=body)

        held_t = ""
        held_s = ""

        def apply_keys(thr: str, str_: str):
            nonlocal held_t, held_s
            t_map = {"accelerate": "ArrowUp", "brake": "ArrowDown", "": None}
            s_map = {"left": "ArrowLeft", "right": "ArrowRight", "": None}
            if thr != held_t:
                if t_map[held_t]:
                    page.keyboard.up(t_map[held_t])
                if t_map[thr]:
                    page.keyboard.down(t_map[thr])
                held_t = thr
            if str_ != held_s:
                if s_map[held_s]:
                    page.keyboard.up(s_map[held_s])
                if s_map[str_]:
                    page.keyboard.down(s_map[str_])
                held_s = str_

        try:
            page.route("**/*editions.summer2025_.drive-*.js", handle_route)
            page.goto(GAME_URL, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_selector('xpath=//span[text()="Start"]', timeout=30000)
            page.wait_for_function(
                "() => window.__stig && window.__stig.K.getState().hasShadersCompiled",
                timeout=30000,
            )

            for lap_idx, mode in enumerate(modes):
                lap_dir = session_dir / f"lap_{lap_idx}_{mode}"
                lap_dir.mkdir(parents=True, exist_ok=True)
                apply_keys("", "")

                # Ensure we are on SplashScreen then click Start.
                scr = page.evaluate("() => window.__stig.K.getState().gameScreen")
                if scr != 0:
                    page.keyboard.press("Escape")
                    time.sleep(0.5)
                page.click('xpath=//span[text()="Start"]')
                print(f"[lap {lap_idx} mode={mode}] started, waiting for countdown...", flush=True)

                # Wait for raceStartTime !== null
                t_wait = time.time()
                while time.time() - t_wait < 15:
                    st = page.evaluate(ORACLE_JS)
                    if st and st.get("ready") and st.get("raceStartTime") is not None:
                        break
                    time.sleep(0.05)

                t_lap = time.time()
                saved = 0
                cols = 0
                steps = 0

                # Perturbation state for "perturbed" mode
                next_perturb = time.time() + rng.uniform(2.5, 4.0)
                perturb_end = 0.0
                perturb_dir = ""

                # Rescue state for "dagger" mode
                stuck_ticks = 0
                rescue_end = 0.0

                while time.time() - t_lap < 160:
                    t_step = time.time()
                    # Capture screenshot first so ORACLE_JS -> apply_keys has minimal latency
                    # and the recorded frame matches the exact state seen before the action.
                    img_bytes = capture_frame()
                    st = page.evaluate(ORACLE_JS)
                    if not st or not st.get("ready"):
                        time.sleep(0.05)
                        continue
                    if st["raceEndTime"] is not None:
                        apply_keys("", "")
                        dur = (st["raceEndTime"] - st["raceStartTime"]) / 1000.0
                        print(
                            f"[lap {lap_idx} mode={mode}] FINISHED in {dur:.2f}s "
                            f"(saved={saved}, cols={cols}/{steps})",
                            flush=True,
                        )
                        break

                    if st["collision"] != -1:
                        cols += 1
                    steps += 1

                    exp_t, exp_s = oracle_action(st)
                    now = time.time()

                    record_this = True
                    exec_t, exec_s = exp_t, exp_s

                    if mode == "perturbed":
                        if now < perturb_end:
                            exec_t, exec_s = "accelerate", perturb_dir
                            record_this = False
                        elif (
                            now >= next_perturb
                            and not st["isInJump"]
                            and 0.015 < st["progress"] < 0.94
                            and abs(st["lat"]) < 3.5
                        ):
                            perturb_dir = rng.choice(["left", "right"])
                            perturb_end = now + rng.uniform(0.35, 0.70)
                            next_perturb = perturb_end + rng.uniform(2.8, 4.8)
                            exec_t, exec_s = "accelerate", perturb_dir
                            record_this = False
                    elif mode == "dagger":
                        mod_t, mod_s = query_agent(agent_url, img_bytes)
                        if st["collision"] != -1 or abs(st["lat"]) > 6.2:
                            stuck_ticks += 1
                        else:
                            stuck_ticks = 0
                        if stuck_ticks >= 6:
                            rescue_end = now + 0.8
                            stuck_ticks = 0
                        if now < rescue_end:
                            exec_t, exec_s = exp_t, exp_s
                        else:
                            exec_t, exec_s = mod_t, mod_s
                        record_this = True

                    if record_this:
                        ts = time.time_ns()
                        fname = f"frame_{ts}_{exp_t}_{exp_s}.jpeg"
                        (lap_dir / fname).write_bytes(img_bytes)
                        saved += 1

                    apply_keys(exec_t, exec_s)

                    elapsed = time.time() - t_step
                    if elapsed < interval:
                        time.sleep(interval - elapsed)

                apply_keys("", "")
                time.sleep(1.0)
        finally:
            apply_keys("", "")
            page.close()


def main():
    parser = argparse.ArgumentParser(description="Collect Horizon Drive training laps.")
    parser.add_argument(
        "--cdp-url",
        default=os.getenv("CDP_URL", "http://127.0.0.1:9222"),
        help="Chrome DevTools HTTP or WS URL",
    )
    parser.add_argument(
        "--recordings-dir",
        default=os.getenv("RECORDINGS_DIR", "assets/recordings"),
        help="Directory to write recorded frames",
    )
    parser.add_argument(
        "--modes",
        default=os.getenv("COLLECT_MODES", "clean,perturbed,perturbed,perturbed"),
        help="Comma-separated list of lap modes: clean, perturbed, dagger",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=int(os.getenv("FRAMES_PER_SECOND", "10")),
    )
    parser.add_argument(
        "--agent-url",
        default=os.getenv("AGENT_URL", "http://localhost:8080"),
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    collect(
        cdp_url=args.cdp_url,
        recordings_dir=args.recordings_dir,
        modes=modes,
        fps=args.fps,
        agent_url=args.agent_url,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
