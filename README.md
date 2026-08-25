# DL-Playground: A Visual Pytorch Deep Learning Prototyping Tool
<h3>
  Built in collaboration with 
  <a href="https://dsgiitr.in/" target="_blank">
    <img src="https://github.com/user-attachments/assets/78e8c579-2b32-4806-9601-753fd786c5e5" width="45" style="vertical-align: middle;" alt="DSG Logo">
  </a>
  &nbsp;×&nbsp;
  <a href="https://sdslabs.co/" target="_blank">
    <img src="https://github.com/user-attachments/assets/8b366d57-6f39-4e91-a065-a83dfceed571" width="70" style="vertical-align: middle;" alt="SDSLabs Logo">
  </a>
</h3>
<br>

**DL-Playground is an interactive, web-based visual editor for designing, prototyping, and understanding PyTorch neural network architectures.**

This tool provides an intuitive drag-and-drop interface that empowers developers and learners to build complex deep learning models without writing code from scratch. See your architecture come to life, from individual layers to complete computational graphs, and instantly generate the corresponding Python code.

## Watch Demo
<a href="https://www.youtube.com/watch?v=fR5L05nidVM">
<img src="https://github.com/user-attachments/assets/3499f9df-f0ed-49a7-ac69-0bf1d80318c6" alt="DL-Playground Demo" width="100%">
</a>

---

## Getting Started

To get the application running locally, follow these steps.

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DL-Playground.git
cd DL-Playground
```

### 2. Run the setup script (Docker required)

```bash
chmod +x setup.sh
./setup.sh
```

The frontend will be available at `http://localhost:7000`. Open this URL in your browser to start using the playground!

---

## Local Development (Run Without Docker Compose)

Run the frontend and backend directly. On Windows use **PowerShell** (not `bash` — the `bash` on this machine resolves to WSL, which cannot reach Windows `localhost`).

### 1. Frontend (Vite dev server)

```powershell
cd frontend
npm install          # first time only
npm run dev          # if npm on PATH is broken, use: D:\nodejs\npm.cmd run dev
```

Runs at `http://localhost:5173`. It calls the training backend at `http://localhost:8000`.

### 2. Backend — DI-engine training service (MVP)

```powershell
conda activate dlbackend          # py3.11 env with fastapi / uvicorn
cd backend
uvicorn training_service:app --host 0.0.0.0 --port 8000
```

`training_service:app` is the standalone RL-training service (no Docker needed); it spawns training workers with `$DING_PYTHON`. Verify it's up:

```powershell
curl.exe http://localhost:8000/api/training/no_such_task/status   # expect 404
```

Acceptance test (PowerShell version — **not** `bash test_training.sh`, which breaks under WSL):

```powershell
powershell -ExecutionPolicy Bypass -File .\test_training.ps1
```

### 3. Backend — original TorchLens trace service (optional)

```powershell
conda activate dlbackend
cd backend
uvicorn runner:app --host 0.0.0.0 --port 8000   # needs Docker daemon
```

The runner spawns a one-shot Docker container per trace, so build the worker image once first:

```powershell
docker build -t torchlens-worker:latest ./backend/
```

### 4. DI-engine training env (`DING_PYTHON`)

The training service shells out to a **separate DI-engine interpreter** — the `dlbackend` process never imports `ding` directly. Its path is read from `DING_PYTHON` and is **never hardcoded**.

| Platform    | Default                                          |
|-------------|--------------------------------------------------|
| Windows     | `D:\anaconda3\envs\ding_env\python.exe`          |
| Linux/macOS | `export DING_PYTHON=/path/to/ding_env/bin/python` |

> ⚠️ This machine has two Anaconda installs. `ding_env` (DI-engine, py3.10) lives under `D:\anaconda3\envs\ding_env`; the backend uses the separate `dlbackend` (py3.11) env.

### 5. Browser workflow (MVP)

1. Start backend (step 2) and frontend (step 1).
2. Open `http://localhost:5173`.
3. Drag an MLP: Input (set feature dim to `4`) → Linear(4→32) → ReLU → Linear(32→2).
4. Open the sidebar **Training** panel → PPO + CartPole → **Start Training**.
5. Watch the reward curve → **Download Model** when done (~1–2 min).

---

## Key Features

- **Visual Graph Editor**: Built with React Flow, allowing for intuitive drag-and-drop construction of models. Connect, arrange, and configure layers with ease.
- **Extensive Node Library**: A rich collection of PyTorch layers and operations, including:
  - **Core Layers**: Linear, Activations (ReLU, Sigmoid, etc.), Softmax, Flatten, Reshape.
  - **Convolutional/Vision**: Conv2d, MaxPool2d, AvgPool2d, Upsample, Residual Blocks.
  - **Recurrent/Sequence**: LSTM, GRU, RNN, Multi-head Attention, Embeddings.
  - **Losses & Metrics**: MSELoss, CrossEntropyLoss, BCELoss, Accuracy.
- **Real-time Code Generation**: Instantly compiles the visual graph into a clean, readable, and copy-pasteable PyTorch `nn.Module` class, complete with `__init__` and `forward` methods.
- **Modular & Reusable Components**: Encapsulate parts of your graph into custom modules and reuse them throughout your architecture, promoting a clean and organized workflow.
- **Live Shape Inference**: Calculates and displays tensor shapes as you build, helping to debug shape-related errors before they happen.
- **Model Inspection**: Leverage the integrated `torchlens` backend to analyze and visualize model summaries and internal states.
- **Built-in Code Editor**: A Monaco-powered editor to view and refine the generated Python code directly in the application.
- **Export to Image**: Export your final graph visualization as a PNG image for documentation or presentations.

---

## Technology Stack

- **Frontend**:
  - **Framework**: React
  - **Language**: TypeScript
  - **Graph Visualization**: React Flow
- **Backend**:
  - **Framework**: FastAPI
  - **Language**: Python
  - **Deep Learning**: PyTorch
  - **Model Analysis**: TorchLens

---

## How to Use

1. **Open the Application**: Navigate to the frontend URL in your browser.
2. **Add Nodes**: Drag layers and operations from the sidebar onto the canvas.
3. **Configure Nodes**: Click on a node to adjust parameters like kernel size, output features, etc.
4. **Connect Nodes**: Drag from the output handle of one node to the input handle of another to create a connection.
5. **View Generated Code**: As you build the graph, the corresponding PyTorch code will automatically update in the "Code" panel.
6. **Create Modules**: Select a group of nodes using <kbd>Shift + L-Click</kbd> and save a module to create your own building block.
7. **Export**: Use the export buttons to save a SVG image of your graph or copy the generated Python code.

---

## Contributing

This project is a combined effort from the DSG Club and SDSLabs at IIT Roorkee.

Contributions are welcome! If you have ideas for new features, bug fixes, or improvements, please feel free to open an issue or submit a pull request. See [CONTRIBUTING.md](./CONTRIBUTING.md) for more details.
