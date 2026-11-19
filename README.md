# Vision Transformer & DeiT — Learning Studio

> Plateforme pédagogique interactive et laboratoire d'ingénierie en Deep Learning & Vision par Ordinateur.

Ce studio académique explore en profondeur le fonctionnement mathématique et pratique des architectures **Vision Transformer (ViT)** (*Dosovitskiy et al., 2020*) et **Data-efficient Image Transformer (DeiT)** (*Touvron et al., 2021*), depuis le prétraitement de l'image brute jusqu'à la classification finale.

---

## 1. Arborescence du Projet

```text
Vision_Transformer_Lab/
├── backend/
│   ├── __init__.py
│   ├── main.py                     # API FastAPI avec hooks PyTorch, tenseurs réels & eager attention
│   ├── test_api_and_math.py        # Suite de tests Pytest (vérification numérique formules == PyTorch)
│   └── Dockerfile                  # Conteneurisation Backend
├── public/
│   └── presets/                    # Galerie académique (bluejay, cat, car, kingfisher, samoyed, espresso, shark)
├── src/
│   ├── api.ts                      # Client API typé TypeScript pour tous les endpoints REST
│   ├── App.tsx                     # Routage principal & vérification de santé backend
│   ├── main.tsx                    # Point d'entrée React, KaTeX CSS & React Router
│   ├── styles.css                  # Design System complet (Marine/Bleu/Violet, responsive & print)
│   ├── components/
│   │   ├── Header.tsx              # Barre de statut, fil d'Ariane, bascule de thème clair/sombre, XP
│   │   ├── Sidebar.tsx             # Navigation principale (Accueil, Comprendre, Calculs, Pratiquer, Comparer, À propos)
│   │   └── MathFormula.tsx         # Rendu KaTeX des formules LaTeX (InlineMath, BlockMath, TeX)
│   ├── pages/
│   │   ├── Home.tsx                # Accueil : sélection d'image active, 3 modes, badges & progression
│   │   ├── About.tsx               # À propos : bibliographie (ICLR/ICML), biais inductif & certificat imprimable
│   │   ├── comprehension/
│   │   │   ├── ArchitectureDiagram.tsx # Schéma interactif en 3 colonnes, tiroir de détails & quiz
│   │   │   ├── TheoreticalCalculations.tsx # Laboratoire mathématique 7 étapes (ViT & DeiT, miniature & réel)
│   │   │   └── lab/
│   │   │       ├── mathEngine.ts   # Moteur arithmétique (LayerNorm, QKV, GELU, Softmax, tolérance)
│   │   │       └── types.ts        # Types TypeScript du laboratoire théorique
│   │   ├── practice/
│   │   │   └── PracticeWorkflow.tsx # Laboratoire pratique en 7 étapes sur les vrais tenseurs PyTorch
│   │   └── comparison/
│   │       └── ModelComparisonLab.tsx # Banc d'essai réel jusqu'à 15 images (top-5, temps, accord, CSV/PDF)
│   └── store/
│       └── useAppStore.ts          # Store Zustand persistant (image active, modèles, scores, badges)
├── docker-compose.yml              # Orchestration multi-conteneurs (FastAPI + Vite/Nginx)
├── Dockerfile.frontend             # Conteneurisation Frontend React
├── nginx.conf                      # Configuration serveur Nginx
├── package.json                    # Dépendances Node.js (React 18, Vite, KaTeX, Zustand, Lucide)
├── requirements.txt                # Dépendances Python (FastAPI, PyTorch, Transformers, Pillow)
└── tsconfig.json                   # Configuration TypeScript stricte
```

---

## 2. Démarrage Rapide

### A. Avec Docker Compose (Recommandé)

```bash
docker-compose up --build
```
- **Frontend :** [http://localhost:5173](http://localhost:5173)
- **Backend FastAPI :** [http://localhost:8000](http://localhost:8000)
- **Swagger Documentation :** [http://localhost:8000/docs](http://localhost:8000/docs)

---

### B. Lancement Manuel en Local

#### 1. Backend (FastAPI + PyTorch)
```powershell
# Créer et activer l'environnement virtuel Python
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

# Installer les dépendances
pip install --upgrade pip
pip install -r requirements.txt
pip install pytest httpx

# Lancer l'API FastAPI
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

#### 2. Frontend (React + Vite + TypeScript)
```powershell
# Dans un second terminal à la racine :
npm install
npm run dev
```
Accédez ensuite à **`http://localhost:5173`**.

---

## 3. Spécification des Endpoints de l'API Backend

| Méthode | Route | Description | Entrée / Paramètres | Sortie Clé |
| :--- | :--- | :--- | :--- | :--- |
| `GET` | `/api/health` | Vérification de statut et GPU | Aucune | `{"status": "ok", "cuda_available": bool}` |
| `GET` | `/api/model-info` | Constantes d'architecture | Aucune | Hyperparamètres ViT et DeiT ($P=16, D=768, L=12, h=12$) |
| `POST` | `/api/preprocess` | Prétraitement réel ImageNet | `file`, `mode` (`resize_crop_256` / `direct_224`), `patch_idx` | Previews colorés RGB, histogrammes, stats de normalisation |
| `POST` | `/api/predict` | Classification Top-5 ViT & DeiT | `files` (1 à 20 images) | Classes prédites, confiance, temps d'inférence en ms, accord |
| `POST` | `/api/attention` | Extraction de matrices d'attention | `file`, `model_name`, `layer` (1..12), `head` (0..12) | Cartes spatiales 14×14 CLS et DIST, échantillons |
| `POST` | `/api/attention-rollout` | Calcul du flux d'attention complet | `file`, `model_name`, `head_fusion` (`mean`/`max`/`min`) | Carte 14×14 d'attention rollout (Abnar & Zuidema, 2020) |
| `POST` | `/api/tensors/intermediate` | Tenseurs et filtres internes | `file`, `model_name`, `patch_idx` | Filtres appris de convolution 16×16, similarité cosinus 2D des positions, normes des 12 couches, logits des deux têtes DeiT |
| `POST` | `/api/perturbation` | Test de robustesse (Mode Défi) | `file`, `perturb_type` (`mask_center`/`mask_random`/`shuffle`) | Prédiction originale vs perturbée, chute de confiance |

---

## 4. Validation & Jeux de Tests Numériques

Une suite de tests Pytest vérifie rigoureusement l'adéquation arithmétique entre les calculs manuels demandés à l'étudiant et les tenseurs PyTorch réels :

```powershell
.venv\Scripts\python.exe -m pytest backend/test_api_and_math.py -v
```

Tests validés :
1. `test_health_endpoint` : Disponibilité de l'API et chargement des identifiants HuggingFace.
2. `test_model_info` : Cohérence des dimensions théoriques ($N=196, D=768, 197/198$ tokens).
3. `test_mathematical_consistency_patch_slicing` : Vérification de $N = (H \cdot W)/P^2$ et $D = P^2 \cdot C$.
4. `test_mathematical_consistency_layer_norm` : Adéquation exacte du calcul manuel de $\mu$, $\sigma^2$ et $\hat{x}$ avec `torch.nn.functional.layer_norm`.
5. `test_mathematical_consistency_attention` : Vérification du produit matriciel $Q K^T$, de l'échelle $\sqrt{d_k} = \sqrt{4} = 2.0$ et du Softmax ligne par ligne.
6. `test_mathematical_consistency_gelu` : Conformité de l'implémentation continue $\text{GELU}(x) = x \Phi(x)$ avec `torch.nn.functional.gelu`.

---

## 5. Publications de Référence

- **ViT :** Dosovitskiy et al. (2020), *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*, ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)
- **DeiT :** Touvron et al. (2021), *Training data-efficient image transformers & distillation through attention*, ICML 2021. [arXiv:2012.12877](https://arxiv.org/abs/2012.12877)
- **Attention Rollout :** Abnar & Zuidema (2020), *Quantifying Attention Flow in Transformers*, ACL 2020. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)
