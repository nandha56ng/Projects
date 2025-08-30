# Continual Learning with Attention-Driven Feature Adaptation and Memory-Based Analytic Classifier

This repository contains the **research implementation of FAAD-Mem**  
(Feature-Adapted Attention-Driven Analytic Learning with Memory Augmentation and Knowledge Distillation).  

FAAD-Mem is designed for **class-incremental continual learning (CIL)** and is currently under review.  

---

## 🔍 Overview
Continual learning aims to acquire new tasks without forgetting previously learned knowledge.  
The main challenge is **catastrophic forgetting**, where new training overwrites old knowledge.  

FAAD-Mem addresses this problem through:
- **Frozen CNN backbone** for stable feature extraction  
- **Feature adaptation + attention module** for re-alignment  
- **Recursive analytic classifier** solved in closed form  
- **Prototype-based memory bank** (exemplar-free replay)  
- **EMA teacher distillation** with cosine-annealed scheduling  
- **In-loop bias correction** to mitigate prediction drift  

---

## 📂 Repository Structure

├── config.py # Experiment arguments
├── main.py # Training script (base, re-align, incremental)
├── FAADMem.py # FAAD-Mem core model
├── datasets/ # Dataset loaders
├── results/ # Output logs (IL.csv, base_training.csv)
└── README.md # Documentation


---

## ⚙️ Installation

We recommend **Python 3.8+** and **PyTorch ≥ 1.11**.

```bash
git clone https://github.com/yourusername/FAAD-Mem.git
cd FAAD-Mem
conda create -n faadmem python=3.8
conda activate faadmem
pip install -r requirements.txt

