v1.0.0 (Production‑Safe Model Management)
----------------------------------------

Features
- Unified Tailwind/Alpine layout with responsive sidebar and dark mode
- Public user page: clean TTS form with auto‑trim + autoplay
- Admin: model list, multiple model versions, activation, per‑model delete
- Admin: download (async), upload, import (Coqui/URL)
- Persistent config (config.json) + env overrides
- Health endpoint enriched (active_models, models_present)

Production Safety
- No auto‑download of models; explicit admin actions only
- Clear errors when no model installed (admin banner + synth JSON 400)
- Thread‑safe model load/reset with global lock
- Hardened archive extraction (no traversal, no symlinks)
- Full cleanup of TTS/Vocoder folders under ./models

CI/CD
- Elastic Beanstalk GitHub Actions workflow kept
- Docs updated for AWS deploy + persistence guidance

