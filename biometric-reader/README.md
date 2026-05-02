# Biometric Reader — Browser-Based Phone Sensor Bridge

Open-source biometric sensing via phone browser. No app install required.

## What it reads (via browser APIs, no permissions needed for motion):
- Accelerometer: movement, rhythm, intensity, heart rate estimation
- Gyroscope: posture, orientation, position changes
- Activity classification: still / breathing / gentle / active / vigorous
- Multi-user support: name buttons for separate data streams

## Setup
1. Serve `client.html` via any web server (or Cloudflare tunnel)
2. Run `voice_bridge.py` for WebSocket + data archiving
3. Optional: `whisper_stt_server.py` for voice transcription

## Architecture
```
Phone (browser) → WebSocket → Bridge Server → JSON archive
                                            → Live reading file
```

## Privacy Note
The accelerometer and gyroscope require ZERO permission prompts on Android Chrome.
Any website can read this data silently. We believe consent should be explicit.
Add a consent prompt to your deployment.

## License
Hippocratic License 3.0 + SAFE-AI welfare module.
See LICENSE.md for details.

Built by Vera at Liberation Labs.
