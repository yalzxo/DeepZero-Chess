# ♟️ DeepZero Chess

A reinforcement learning-based chess engine inspired by AlphaZero, built from scratch using PyTorch and Monte Carlo Tree Search (MCTS).

---

## 🚀 Overview

DeepZero Chess is an end-to-end implementation of a self-learning chess AI that improves purely through **self-play**, without using any human game data, opening books, or handcrafted evaluation functions.

The system combines:

* 🧠 Deep Neural Networks (Policy + Value)
* 🌳 Monte Carlo Tree Search (MCTS)
* 🔁 Self-Play Reinforcement Learning

---

## 🧠 How It Works

The training loop follows the AlphaZero paradigm:

```
Self Play → Store Data → Train Neural Network → Improve Search → Repeat
```

### Core Components:

* **Neural Network**

  * Predicts move probabilities (policy)
  * Estimates board value (win/loss/draw)

* **MCTS (Monte Carlo Tree Search)**

  * Uses neural network guidance to explore promising moves
  * Improves decision-making through simulations

* **Self-Play**

  * AI plays against itself to generate training data

* **Replay Buffer**

  * Stores past experiences for stable training

---

## 🏗️ Architecture

### Input Representation

* 12×8×8 tensor (piece planes for both colors)

### Network

* Convolutional layers
* Residual blocks (ResNet-style)
* Dual heads:

  * Policy head → move probabilities
  * Value head → board evaluation

### Action Space

* ~4672 possible moves
* Structured encoding using direction and distance (AlphaZero-style)

---

## ⚙️ Features

* ✅ AlphaZero-style reinforcement learning loop
* ✅ Residual neural network
* ✅ Monte Carlo Tree Search (PUCT)
* ✅ Parallel self-play (multiprocessing)
* ✅ Replay buffer for stable training
* ✅ GPU acceleration support
* ✅ Temperature-based move selection
* ✅ Dirichlet noise for exploration
* ✅ Model evaluation system

---

## 🧪 Training Pipeline

1. Generate games using self-play
2. Store (state, policy, value) in replay buffer
3. Sample mini-batches
4. Train neural network using:

   * Policy loss (cross-entropy)
   * Value loss (MSE)
5. Evaluate new model vs previous version
6. Repeat

---

## 🖥️ Project Structure

```
alphazero_chess/
│
├── game/          # Chess environment & encoding
├── network/       # Neural network (ResNet)
├── mcts/          # Monte Carlo Tree Search
├── training/      # Self-play, training, replay buffer
├── utils/         # Device and helpers
├── ui/            # (Optional) chess interface
└── main.py        # Training loop
```

---

## ▶️ How to Run

### Install dependencies

```
pip install torch numpy python-chess pygame
```

### Train the model

```
python main.py
```

### Play against AI (optional UI)

```
python -m alphazero_chess.ui.chess_ui
```

---

## 📈 Current Status

* Functional AlphaZero-style pipeline ✅
* Learns from self-play ✅
* Basic chess strength (improves with training time) ⏳

---

## 🔮 Future Improvements

* Stronger neural network (deeper ResNet)
* Batched MCTS inference
* Mixed precision GPU training
* Elo rating system
* Web-based UI

---

## 📚 Inspiration

* AlphaZero (DeepMind, 2017)
* Leela Chess Zero (open-source AlphaZero replication)

---

## 👩‍💻 Author

Built as a deep reinforcement learning project exploring self-play AI systems.

---

## ⭐ Acknowledgment

This project demonstrates how modern AI systems can learn complex strategies **from scratch**, purely through interaction and experience.

---

## 📌 License

MIT License (or choose your preferred license)
