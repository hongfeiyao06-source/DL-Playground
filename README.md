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

For day-to-day development you can run the frontend and backend directly.

### 1. Frontend (Vite dev server)

```bash
cd frontend
npm install
npm run dev
```

The dev server runs at Vite's default `http://localhost:5173`. It talks to the backend at `http://localhost:8000` (see `frontend/src/utils/traceService.ts`).

### 2. Backend (FastAPI + TorchLens worker)

```bash
conda activate dlbackend          # py3.11 env with fastapi / uvicorn / docker
cd backend
uvicorn runner:app --host 0.0.0.0 --port 8000
```

The runner spawns a one-shot Docker container per trace, so build the worker image once first:

```bash
docker build -t torchlens-worker:latest ./backend/
```

### 3. DI-engine training service (`DING_PYTHON`)

The RL training backend shells out to a **separate DI-engine interpreter** — the `dlbackend` process never imports `ding` directly. Its path is read from the `DING_PYTHON` environment variable and is **never hardcoded**.

| Platform    | Default                                          |
|-------------|--------------------------------------------------|
| Windows     | `D:\anaconda3\envs\ding_env\python.exe`          |
| Linux/macOS | `export DING_PYTHON=/path/to/ding_env/bin/python` |

> ⚠️ This machine has two Anaconda installs. `ding_env` (DI-engine, py3.10) lives under `D:\anaconda3\envs\ding_env`; the backend uses the separate `dlbackend` (py3.11) env.

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
