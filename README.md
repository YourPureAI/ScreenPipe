<p align="center">
  <img src="https://img.shields.io/badge/YourPureAI-ScreenPipe-black?style=for-the-badge&logo=rust" alt="logo">
</p>

<h1 align="center">YourPureAI ScreenPipe (Distributed Edition)</h1>

<p align="center"><b>AI memory for your screen, optimized for resource-constrained devices.</b></p>
<p align="center">A high-performance fork introducing a Distributed Capture-Processor Architecture.</p>

---

## What is this?

YourPureAI ScreenPipe turns your computers into a personal AI that knows everything you've done. 
Unlike standard recording solutions that bog down your CPU and RAM by running heavy AI models continuously, this **distributed edition** splits the workload perfectly across your network.

```text
┌─────────────────────────────────┐           ┌────────────────────────────────────┐
│ Thin Client (Your Laptop)       │           │ Processing Server (e.g. Mac Mini)  │
│                                 │           │                                    │
│ 1. Capture screen & audio       │   Queue   │ 1. Ingest raw packets              │
│ 2. Change detection             ├──────────►│ 2. Run Whisper & OCR models        │
│ 3. Save raw HD packets (.bin)   │           │ 3. Save to database / search AI    │
└─────────────────────────────────┘           └────────────────────────────────────┘
```

- **Hyper-Lightweight Client:** Run the thin client on weak Windows/Linux laptops. It bypasses OCR and Whisper entirely, preserving your battery and CPU (uses just ~100MB RAM and 3-10% CPU).
- **Heavy Remote Processing:** Offload all the AI crunching (computer vision, voice transcription) to a powerful centralized machine over your local network. Timestamps stay absolutely accurate to the original capture moments.
- **Still 100% Local:** Your data never touches the cloud. It moves seamlessly across a private network folder from your client machine to your own local processing server.

## Installation & Usage

You can run this project in three modes using the new `--mode` flag. 

### 1. The Thin Client (Capture Only)
Run this on your everyday laptop. It will capture frames and audio, detect changes, compress images, and push them to a shared `queue-dir`. All heavy AI models are permanently disabled in this mode.
```bash
./screenpipe.exe record --mode capture-only --queue-dir z:\shared\screenpipe-queue
```

### 2. The Processing Server (Process Only)
Run this on your powerful server (e.g., Mac Mini, PC with GPU). It polls the shared queue directory, runs Whisper (speech-to-text) and OCR (image-to-text), and populates your local SQLite database.
```bash
./screenpipe record --mode process-only --queue-dir /Volumes/shared/screenpipe-queue
```
**📥 [Download the latest compiled Processing Server (macOS Apple Silicon)](https://github.com/YourPureAI/ScreenPipe/actions/runs/23941866739)**

### 3. Local Mode (Original Behavior)
Run everything on a single, powerful machine in an all-in-one loop.
```bash
./screenpipe record --mode local
# or simply:
./screenpipe record
```

### Advanced: Using Local LLMs (LM Studio / Ollama)
If you prefer running your own local AI for transcription instead of the bundled Whisper model, configure the processing server via the CLI. For example, if you run LM Studio locally on port 1234:
```bash
./screenpipe record --mode process-only --queue-dir /Volumes/shared/queue \
  --audio-transcription-engine openai-compatible \
  --openai-compatible-endpoint http://localhost:1234/v1
```

## Core Features

- **Event-Driven Visual Capture:** Smart frame comparison ensures we only save frames when the screen actually changes.
- **Secure Audio Batching:** Atomically saves temporary audio streams to prevent partial network reads on the server side.
- **Zero-Data-Loss Processing:** The processing server strictly inherits timestamps from the initial capture moment, regardless of how long it took to process the queue.
- **Distributed Ready:** Built to work effortlessly over SMB, NFS, Syncthing, or any shared directory protocol.

## Building for Windows (Thin Client)
We provide a dedicated GitHub Actions workflow to auto-build the lightweight Windows client (`Build Windows CLI`).
1. Navigate to your GitHub `Actions` tab.
2. Select the `Build Windows CLI` workflow.
3. Download the zipped artifact containing `screenpipe.exe` and necessary runtime dlls.

**📥 [Download the latest compiled Thin Client (Windows x86_64)](https://github.com/YourPureAI/ScreenPipe/actions/runs/23915655851)**

---

## Acknowledgements

This project is a heavily customized, distributed fork built out of the incredible [Screenpipe](https://github.com/screenpipe/screenpipe) engine by Louis Beaumont and contributors. The core event-driven capturing, UI, and standard pipelines remain heavily inspired and powered by their foundational work. 

If you are looking for the standard all-in-one desktop experience with a graphical user interface offering, please check out the original project.

- **Original Project:** [screenpipe/screenpipe](https://github.com/screenpipe/screenpipe)
- **Original License:** MIT
