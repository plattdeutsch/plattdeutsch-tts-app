Plattdeutsch TTS App
====================

Minimal Flask Web-App für deutsche Text-zu-Sprache (Coqui TTS) mit optionalem Upload nach Amazon S3 und automatischem Deployment via GitHub Actions nach AWS Elastic Beanstalk (eu-central-1).

Funktionen
---------
- Eingabe eines Textes im Browser, Erzeugung einer MP3-Sprachdatei
- MP3 direkt herunterladen oder als Presigned-URL aus S3 erhalten
- Health-Endpoint mit Modellinformationen

Laufzeit-Umgebung
-----------------
- Python 3.10
- Flask 3.0.3, TTS 0.22.0, numpy 1.26.4, lameenc 1.6.1, boto3 1.35.0, gunicorn 22.0.0

Schnellstart (Lokal)
--------------------
1. Python 3.10 installieren
2. Abhängigkeiten installieren und App starten:

   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\\Scripts\\Activate.ps1
   # macOS/Linux
   # source .venv/bin/activate

   pip install -r requirements.txt
   python app.py
   ```

3. Browser öffnen: http://localhost:5000

Konfiguration (Umgebung)
------------------------
Die wichtigsten Variablen werden in `.ebextensions/01_env.config` für Elastic Beanstalk gesetzt und können auch lokal als Umgebungsvariablen überschrieben werden:

- `AWS_REGION` (Standard: `eu-central-1`)
- `TTS_MODEL_NAME` (Standard: `tts_models/de/thorsten/tacotron2-DDC`)
- `USE_S3` (`true`/`false`, Standard: `true`)
- `S3_BUCKET_NAME` (Standard: `plattdeutsch-tts-audio`)

Health Check
------------
`GET /health` liefert JSON mit `model_name`, `model_loaded`, `sample_rate`, `use_s3`, `bucket`, `region`.

Deployment nach Elastic Beanstalk
---------------------------------
GitHub Actions Workflow: `.github/workflows/deploy-eb.yml`

Voraussetzungen:
- EB Application: `plattdeutsch-tts-app`
- EB Environment: `plattdeutsch-tts-env` (Python 3.10 Plattform)
- GitHub Secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`

Ablauf:
- Push auf `main` baut ein Zip und deployed über `einaregilsson/beanstalk-deploy` nach EB.

Hinweise
--------
- ffmpeg wird nicht verwendet; MP3-Encoding erfolgt mit `lameenc` in-memory.
- Das Laden des Coqui-Modells erfolgt lazy beim ersten Synthese-Aufruf.
- Bei aktivem S3-Upload wird ein Presigned-Download-Link (1 Stunde) zurückgegeben.

