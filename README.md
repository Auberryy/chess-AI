# ChessAI — AlphaZero-style Training System ✅

[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)

A high-performance research implementation of an AlphaZero-style chess training system. Written in C++ with optional CUDA acceleration and LibTorch (PyTorch C++ API), the project focuses on reproducible self-play training, efficient MCTS, and evaluation against Stockfish.

---

## Quickstart 🔧

Prerequisites:
- CMake 3.18+
- C++17 compiler (GCC/Clang/MSVC)
- CUDA 11.0+ (optional — required for GPU acceleration)
- LibTorch (PyTorch C++ API)
- Stockfish (optional, for evaluation)

Clone and build:

```bash
git clone <repository-url>
cd ChessAI
# Prepare LibTorch and optional CUDA
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="/path/to/libtorch"
cmake --build . --config Release
```

Run training (example):

```bash
# Train on GPU 0, use 50% GPU memory
./chess_ai --train --gpu 0 --gpu-memory 0.5
```

See more detailed build and platform-specific instructions in the "Build" section below.

---

## Features ✨

- AlphaZero-style deep residual network with policy and value heads
- Parallel MCTS optimized for batched inference
- Self-play training loop with checkpointing
- Elo-based challenger/champion promotion system
- Optional NNUE training for CPU inference
- Stockfish evaluation harness and automated model promotion
- Monitoring UI and resource panels for training diagnostics

---

## Building (detailed) 🔨

### Windows (Visual Studio)

1. Download LibTorch (CUDA or CPU).
2. Extract to a known path (e.g. `C:/libtorch`).
3. Configure and build:

```powershell
mkdir build
cd build
cmake .. -DCMAKE_PREFIX_PATH="C:/libtorch" -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

### Linux / macOS

Install dev tools and dependencies (example for Ubuntu):

```bash
sudo apt update && sudo apt install -y cmake build-essential libgl1-mesa-dev libxrandr-dev libxinerama-dev libxcursor-dev libxi-dev
# Download and extract LibTorch
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH="$PWD/../libtorch"
make -j$(nproc)
```

Notes:
- Use a LibTorch build matching your CUDA version when enabling GPU support.
- `CMAKE_PREFIX_PATH` should point to the LibTorch root directory.

---

## Usage & Examples 📋

Start training from scratch:

```bash
./chess_ai --train --gpu 0 --gpu-memory 0.5
```

Resume training:

```bash
./chess_ai --resume checkpoints/checkpoint_100.pt
```

Play against a model:

```bash
./chess_ai --play checkpoints/best_model.pt
```

Evaluate vs Stockfish:

```bash
./chess_ai --evaluate checkpoints/best_model.pt --stockfish /path/to/stockfish
```

Common flags:
- `--gpu <id>`: GPU device id (default: 0)
- `--gpu-memory <f>`: Fraction of GPU memory to reserve (0.0–1.0)
- `--use-amp` / `--no-amp`: Enable/disable mixed precision
- `--simulations <n>`: MCTS simulations per move (default: 800)
- `--games <n>`: Self-play games per iteration
- `--elo-training`: Enable challenger/champion promotion
- `--train-nnue`: Train NNUE alongside the main network

For full option list run `./chess_ai --help`.

---

## Elo-based Promotion System 📈

When `--elo-training` is enabled, new models (challengers) must demonstrate an Elo gain against the champion to be promoted. Default evaluation uses 100 games; change with `--eval-games` and `--elo-threshold` for stricter promotion.

---

## Project Structure 🗂️

A simplified view of the repository:

```
ChessAI/
├── include/        # Public headers (core, nn, mcts, training, ui)
├── src/            # Implementation
├── tests/          # Unit and integration tests
├── checkpoints/    # Saved training checkpoints
├── models/         # Saved champion models
├── scripts/        # Utilities and verification scripts
├── CMakeLists.txt
└── README.md
```

---

## Tests & Validation ✅

- Unit tests live under `tests/` (e.g., perft tests).
- Run tests with your preferred CTest or a direct executable produced by the build.

---

## Contributing 🙌

Contributions are welcome — please open issues for bugs or feature requests and submit pull requests for changes.

Guidelines:
- Fork the repository and create a branch per change
- Add or update tests for behavior changes
- Keep PR descriptions concise and link related issues

Please follow a standard git workflow and sign-off on contributions if required by repository policy.

---

## License & Acknowledgments 📜

- **GNU General Public License v3.0** — see `LICENSE` for full text.
- **Note:** This project is released under GPLv3 to maintain compatibility with GPLv3-licensed dependencies (e.g., Stockfish).
- Thanks to DeepMind (AlphaZero), Leela Chess Zero, and Stockfish for inspiration and tooling.

---

## Contact / Support 💬

If you run into issues, please open an issue on GitHub with details (platform, build commands, logs). For quick questions, include reproduction steps and config snippets.

---

Happy training! ♟️

