# Neuro-Dreamer
<p align="center">
  <img src="Demo-Video/Mario-Progression-Run.gif" width="700">
</p>

### Egocentric Foveated Perception for Dreamer in Super Mario Bros
## Overview
Neuro-Dreamer explores how **perceptual bias affects model-based reinforcement learning**.

This project compares two agents:

1. **Baseline Dreamer** using standard pixel observations
2. **Modified Dreamer with egocentric foveated perception**

The modified agent receives a **Gaussian-blurred observation**, where visual clarity is concentrated around the agent's location using `(xpos, ypos)` information from the environment.

This mimics **biological foveated vision**, where perception is sharp near the center of attention and blurred in the periphery.

The goal of the experiment is to evaluate whether **egocentric perception improves representation learning and policy learning in Dreamer agents**.

---

## Research Idea

Biological vision systems do not process every pixel with equal importance.
Humans rely on **foveated perception**, where the central region is high-resolution while peripheral vision is lower resolution.

Inspired by this principle, this project modifies the Dreamer observation pipeline by applying **spatially-aware Gaussian blur** centered on the agent's position.

This creates an **egocentric visual representation** that prioritizes local information.

---

## Experiment Setup

Two agents are trained and compared:

### Baseline Dreamer

* Standard visual input
* Raw environment frames
* No perceptual modification

### Neuro-Dreamer (Foveated)

* Observations blurred using **Gaussian blur**
* Blur intensity increases with distance from `(xpos, ypos)`
* Central region remains sharp
* Peripheral regions progressively blurred

This forces the model to prioritize **locally relevant information**.

---

## Foveated Observation Pipeline

Observation transformation:

```
Environment Frame
        ↓
Extract agent position (xpos, ypos)
        ↓
Apply spatial Gaussian blur
        ↓
Foveated observation
        ↓
Dreamer encoder
```

The encoder therefore learns representations from **attention-biased observations**.

---

## Architecture

The learning algorithm follows the **Dreamer model-based RL architecture**:

Components:

* Visual Encoder
* Recurrent State Space Model (RSSM)
* Reward Model
* Actor Network
* Critic Network

Training occurs using **latent imagination rollouts**.

---

## Environment

Environment: **Super Mario Bros**

Observations:

* RGB frames

Additional information used:

* Agent `(xpos, ypos)` from environment info

Action space:

* Discrete controller inputs

---

## Installation

Clone the repository:

```bash
git clone https://github.com/DHAANUS/Neuro-Dreamer.git
cd Neuro-Dreamer
```

Install dependencies:

```bash
pip install -r requirements.txt
```
After installing the dependencies, training can be started using:
```bash
python main.py
```
This will initialize the Dreamer training pipeline and begin interaction with the Super Mario Bros environment.

## Evaluation Metrics

Agent performance is evaluated using:

* Episode reward
* Learning speed
* World model reconstruction loss
* KL divergence
* Policy entropy

Comparisons are made between:

* Baseline Dreamer
* Foveated Dreamer

---

## Project Structure

```
Neuro-Dreamer
│
├── Config
│ └── config.yml
│
├── DreamerCore
│ ├── BehaviourModel.py
│ ├── CoreInit.py
│ ├── EnvInteraction.py
│ ├── WorldModel.py
│ ├── SaveCheckPoints.py
│ └── LoadCheckPoints.py
│
├── EnvProperties
│ ├── env.py
│ └── f_env.py # Foveated environment wrapper
│
├── Heads
│ ├── Actor.py
│ ├── Critic.py
│ ├── RewardModel.py
│ └── ContinueModel.py
│
├── Models
│ ├── Encoder.py
│ ├── Decoder.py
│ └── RSSM.py
│
├── Utils
│
├── main.py
│
├── README.md
└── LICENSE
```

---

## Key Idea

Instead of changing the RL algorithm itself, this project studies **how modifying perception affects model-based learning**.

The hypothesis is that **egocentric perception may improve representation learning by removing irrelevant visual information**.

---

## Future Work

Possible extensions:

* Attention Mechanism
* Frozen ViT based Encoder
* Liquid Neural Network for Actor

---
## Acknowledgements

This project builds upon ideas and open-source implementations from the reinforcement learning community, particularly work surrounding the **Dreamer family of model-based reinforcement learning algorithms**.

The Dreamer algorithm was originally introduced by **Danijar Hafner et al.**, who proposed learning policies by imagining trajectories inside a learned world model.

Several open-source implementations were studied and referenced during the development of this project:

* **[NaturalDreamer](https://github.com/InexperiencedMe/NaturalDreamer)** – by InexperiencedMe(This repo was my base reference, I thank this repo author a lot for making Dreamer very much simple and understandable than every other implementations)
* **[dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)** – by NM512
* **[SimpleDreamer](https://github.com/kc-ml2/SimpleDreamer)** – by kc-ml2
* **[dreamer-pytorch](https://github.com/yusukeurakami/dreamer-pytorch)** – by yusukeurakami
* **[DreamerV2](https://github.com/danijar/dreamerv2)** – official implementation by Danijar Hafner
* **[SheepRL](https://github.com/Eclectic-Sheep/sheeprl)** – by Eclectic-Sheep
* **[dreamerv3](https://github.com/naivoder/dreamerv3)** – by naivoder

These repositories were helpful for understanding implementation details such as:

* World model training
* Recurrent State Space Models (RSSM)
* Actor–Critic learning through latent imagination
* Training pipelines and experiment organization

I thank the authors and contributors of these projects for making their work publicly available.

## License

MIT License
