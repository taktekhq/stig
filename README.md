# stig

> The ultimate racer.

A mono-repo for the AI that plays the [Shopify Horizon Drive game][shopify-drive].

https://github.com/user-attachments/assets/584655f4-d168-49b9-a395-c6b7d8b57d83

One full lap driven by the `racer` model, 1:52.517. The keys in the bottom left are what the model is holding.

## Prerequisites

- [Docker](https://www.docker.com/get-started) - Required for running the game client and autopilot.
- [Google Chrome](https://www.google.com/chrome/) - Required for browser automation (other browsers may work but are not documented).

## Quick Start

Follow these steps to get Stig up and running:

### 1. Initial Setup

First, set up the environment configuration files and create necessary directories:

```bash
# Create environment files from templates
make env

# Create asset directories (datasets, models, recordings)
make assets
```

### 2. Download Pre-trained Model

Download the pre-trained racer model:

```bash
make stig-racer
```

It finishes laps on its own in about 1:55. The older `novice` model (`make stig-novice`, then `MODEL_NAME=novice` in `stig/env/.env`) is kept for reference and does not finish a lap.

### 3. Start Chrome with Remote Debugging

Open a new terminal and start Chrome with remote debugging enabled:

**macOS:**
```bash
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222 --user-data-dir=/tmp/stig-profile
```

**Linux:**
```bash
google-chrome --remote-debugging-port=9222 --user-data-dir=/tmp/stig-profile
```

The game client needs about 10 screenshots a second, so Chrome must render WebGL on a GPU. On a machine without a screen, add `--headless=new --enable-gpu --use-angle=gl-egl --mute-audio`. With software WebGL (SwiftShader) a screenshot takes seconds and the car cannot steer.

### 4. Configure WebSocket URL

After starting Chrome, you'll see output like this:
```
DevTools listening on ws://127.0.0.1:9222/devtools/browser/2b1ea713-517a-479e-a17b-3958601b23fb
```

Copy the WebSocket URL (the `ws://...` part) and update your environment files:

1. Edit `game/env/.env`
2. Set `BROWSER_WS_URL` to the WebSocket URL you copied

Example:
```properties
BROWSER_WS_URL=ws://127.0.0.1:9222/devtools/browser/2b1ea713-517a-479e-a17b-3958601b23fb
```

### 5. Run Stig

Open two new terminals and run these commands:

**Terminal 1 - Start the autopilot:**
```bash
make stig-autopilot
```

**Terminal 2 - Start the game client:**
```bash
make game-play
```

### Additional Commands

- **Record human gameplay:** `make game-record`
- **Collect automated oracle / DAgger training laps:** `python -m stig.cmd.collect.main --cdp-url http://127.0.0.1:9222 --recordings-dir assets/recordings --modes clean,perturbed,dagger,dagger` (requires Playwright)
- **Train a new model:** `make stig-train` (trains `MODEL_NAME` from `RECORDINGS_DIR` into `assets/models/<MODEL_NAME>.pt`; default `MODEL_NAME=racer`, `EPOCHS=25`, `BATCH_SIZE=64`)

> Note: `stig/cmd/autopilot` automatically detects whether a loaded `.pt` model expects 1-channel grayscale (`novice.pt`) or 3-channel RGB (`racer.pt`).

[shopify-drive]: https://www.shopify.com/ca/editions/summer2025/drive
