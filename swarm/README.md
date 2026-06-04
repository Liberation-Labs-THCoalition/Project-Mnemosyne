# Swarm — Microagent SDLC Pipeline

RAM-based code review pipeline using small local models (Qwen2.5-Coder 1.5B via Ollama). Pre-filters code changes through lint, review, and security agents before expensive cloud review. Triggered via NATS on `system.pipeline` or manually by file path.

## Files

- **pipeline.py** -- Core pipeline: runs lint, review, and security agents against a file or git diff. Agents query Ollama and return severity-graded findings. Escalates error-level results to Opus for deep review.
- **swarm_memory.py** -- SQLite-backed communal memory for the swarm. Records every finding, tracks false positive patterns, and maintains a hall-of-fame for entertaining hallucinated vulnerabilities.
- **swarm_service.py** -- NATS listener (systemd service). Subscribes to `system.pipeline`, runs the pipeline on incoming requests, publishes results back to NATS, records to swarm memory, and notifies Discord on escalations.
