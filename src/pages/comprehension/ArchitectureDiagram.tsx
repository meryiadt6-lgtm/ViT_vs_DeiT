import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BookOpen,
  Info,
  Lightbulb,
  X,
  ArrowRight,
  Calculator,
  Play,
  HelpCircle,
  CheckCircle2,
  XCircle,
  Award,
  ExternalLink,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { TeX, InlineMath, BlockMath } from '../../components/MathFormula';

interface BlockDetail {
  id: string;
  title: string;
  category: 'token' | 'norm' | 'attention' | 'mlp' | 'head' | 'distill';
  formula: string;
  role: string;
  why: string;
  theoryStepId: number;
  practiceStepId: number;
}

const BLOCK_DETAILS: Record<string, BlockDetail> = {
  patch_embed: {
    id: 'patch_embed',
    title: 'Découpage en patches & Projection linéaire',
    category: 'token',
    formula: '\\mathbf{x}_p \\in \\mathbb{R}^{N \\times (P^2 C)},\\quad \\mathbf{z}_0 = \\mathbf{x}_p \\mathbf{E} + \\mathbf{b},\\quad \\mathbf{E} \\in \\mathbb{R}^{(P^2 C) \\times D}',
    role: 'Transforme une image 2D continue en une séquence 1D de N vecteurs latents de dimension D=768, assimilables à des tokens de mots.',
    why: 'Les Transformers ont une complexité d\'attention quadratique O(N²). Traiter chaque pixel individuellement (224×224 = 50 176 pixels) serait prohibitif. Avec P=16, N=196 tokens, le calcul est fluide.',
    theoryStepId: 2,
    practiceStepId: 2,
  },
  pos_embed: {
    id: 'pos_embed',
    title: 'Encodage Positionnel 1D apprenable',
    category: 'token',
    formula: '\\mathbf{Z} = [\\mathbf{x}_{\\text{class}};\\, \\mathbf{z}_0] + \\mathbf{E}_{\\text{pos}},\\quad \\mathbf{E}_{\\text{pos}} \\in \\mathbb{R}^{(N+1) \\times D}',
    role: 'Injecte l\'information de localisation spatiale dans chaque token.',
    why: 'L\'auto-attention est une opération invariante par permutation. Sans encodage positionnel, le modèle traiterait un sac de patches désordonnés sans savoir quel patch est en haut à gauche ou en bas à droite.',
    theoryStepId: 3,
    practiceStepId: 4,
  },
  layernorm: {
    id: 'layernorm',
    title: 'Layer Normalization (Pre-LN)',
    category: 'norm',
    formula: '\\text{LN}(\\mathbf{x}) = \\frac{\\mathbf{x} - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} \\odot \\boldsymbol{\\gamma} + \\boldsymbol{\\beta},\\quad \\mu = \\frac{1}{D}\\sum_{i=1}^D x_i',
    role: 'Normalise les activations de chaque token indépendamment à moyenne 0 et variance 1 avant l\'attention et le MLP.',
    why: 'Stabilise la dynamique du gradient dans les 12 couches et empêche l\'explosion ou l\'évanouissement des gradients, permettant un entraînement direct sans Batch Normalization.',
    theoryStepId: 4,
    practiceStepId: 6,
  },
  attention: {
    id: 'attention',
    title: 'Multi-Head Self-Attention (MHSA)',
    category: 'attention',
    formula: '\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}}\\right)\\mathbf{V},\\quad \\text{MHSA} = [\\text{head}_1; \\dots; \\text{head}_h]\\mathbf{W}_O',
    role: 'Permet à chaque patch d\'échanger de l\'information avec tous les autres patches de l\'image en calculant des poids d\'affinité contextuelle.',
    why: 'Offre un champ récepteur global dès la première couche : un oiseau peut immédiatement corréler son bec avec ses pattes ou le perchoir distant.',
    theoryStepId: 5,
    practiceStepId: 5,
  },
  mlp: {
    id: 'mlp',
    title: 'MLP / Feed-Forward Network avec GELU',
    category: 'mlp',
    formula: '\\text{MLP}(\\mathbf{x}) = \\text{GELU}(\\mathbf{x}\\mathbf{W}_1 + \\mathbf{b}_1)\\mathbf{W}_2 + \\mathbf{b}_2,\\quad \\mathbf{W}_1 \\in \\mathbb{R}^{D \\times 4D},\\,\\mathbf{W}_2 \\in \\mathbb{R}^{4D \\times D}',
    role: 'Applique une transformation non linéaire ponctuelle token par token avec une expansion de dimension D=768 → 4D=3072 puis contraction.',
    why: 'L\'attention ne fait que mélanger les tokens existants. Le réseau MLP transforme les représentations individuelles et stocke la mémoire factuelle du modèle.',
    theoryStepId: 6,
    practiceStepId: 6,
  },
  cls_token: {
    id: 'cls_token',
    title: 'Token spécial [CLS]',
    category: 'token',
    formula: '\\mathbf{y} = \\text{softmax}(\\mathbf{z}_L^0 \\mathbf{W}_{\\text{head}} + \\mathbf{b})',
    role: 'Vecteur d\'agrégation global sans patch initial associé qui recueille l\'information de toute l\'image pour la classification.',
    why: 'Évite de devoir moyenner arbitrairement tous les patches (Global Average Pooling) et permet au modèle d\'apprendre ce qui caractérise la classe globale.',
    theoryStepId: 7,
    practiceStepId: 7,
  },
  dist_token: {
    id: 'dist_token',
    title: 'Token de distillation [DIST] (DeiT)',
    category: 'distill',
    formula: '\\mathcal{L}_{\\text{global}}^{\\text{hard}} = \\frac{1}{2}\\mathcal{L}_{\\text{CE}}(\\psi(\\mathbf{z}_{\\text{cls}}), y) + \\frac{1}{2}\\mathcal{L}_{\\text{CE}}(\\psi(\\mathbf{z}_{\\text{dist}}), y_{\\text{teacher}})',
    role: 'Token exclusif à DeiT qui interagit par attention avec les patches et reproduit la prédiction du modèle CNN enseignant (RegNetY).',
    why: 'Permet à DeiT de bénéficier du fort biais inductif convolutif (localité spatiale) sans avoir besoin de pré-entraînement sur des centaines de millions d\'images.',
    theoryStepId: 7,
    practiceStepId: 7,
  },
};

export function ArchitectureDiagram() {
  const navigate = useNavigate();
  const { imageUrl, validateTheoryStep } = useAppStore();

  const [activeBlock, setActiveBlock] = useState<BlockDetail | null>(null);

  // Quiz State
  const [quizAnswers, setQuizAnswers] = useState<Record<number, number>>({});
  const [quizSubmitted, setQuizSubmitted] = useState<boolean>(false);

  const quizQuestions = [
    {
      id: 1,
      q: 'Combien de tokens composent la séquence d\'un ViT-Base 16x16 sur une image 224x224 ?',
      options: ['196 tokens', '197 tokens (196 patches + 1 [CLS])', '198 tokens', '768 tokens'],
      correct: 1,
      explanation: '14 × 14 = 196 tokens de patches + 1 token spécial [CLS] concaténé au début = 197 tokens.',
    },
    {
      id: 2,
      q: 'Quelle est la particularité architecturale majeure de DeiT par rapport à ViT ?',
      options: [
        'Il remplace l\'attention par des convolutions',
        'Il utilise 24 couches au lieu de 12',
        'Il ajoute un token [DIST] guidé par un enseignant CNN',
        'Il supprime l\'encodage positionnel',
      ],
      correct: 2,
      explanation: 'DeiT introduit un token de distillation [DIST] guidé par un réseau CNN enseignant (RegNetY).',
    },
    {
      id: 3,
      q: 'Pourquoi divise-t-on le produit scalaire QKᵀ par √d_k dans l\'auto-attention ?',
      options: [
        'Pour éviter que les valeurs ne deviennent trop grandes et saturent le Softmax',
        'Pour inverser la matrice K',
        'Pour accélérer la multiplication matricielle',
        'Pour supprimer le biais',
      ],
      correct: 0,
      explanation: 'Pour des dimensions d_k élevées (64), le produit scalaire croît en variance, repoussant le Softmax dans des zones à gradients évanescents.',
    },
    {
      id: 4,
      q: 'Comment DeiT combine-t-il les prédictions des deux têtes lors de l\'inférence finale ?',
      options: [
        'Il ne garde que la tête CLS',
        'Il prend la moyenne arithmétique des probabilités des deux têtes',
        'Il choisit toujours la tête du CNN enseignant',
        'Il multiplie les logits',
      ],
      correct: 1,
      explanation: 'À l\'inférence, la prédiction finale de DeiT est P_final = 0.5 * (P_cls + P_dist).',
    },
  ];

  const handleSelectAnswer = (qId: number, optIdx: number) => {
    if (quizSubmitted) return;
    setQuizAnswers((prev) => ({ ...prev, [qId]: optIdx }));
  };

  const submitQuiz = () => {
    setQuizSubmitted(true);
    const correctCount = quizQuestions.filter((q) => quizAnswers[q.id] === q.correct).length;
    if (correctCount >= 3) {
      validateTheoryStep(1, 10);
    }
  };

  return (
    <div className="unified-page-container">
      {/* 1. BANNER HEADER */}
      <div className="unified-banner-card">
        <div className="banner-left-content">
          <div className="banner-icon-box">
            <BookOpen size={28} className="banner-main-icon" />
          </div>
          <div className="banner-titles">
            <h1 className="banner-title">Comprendre les architectures ViT et DeiT</h1>
            <p className="banner-subtitle">
              De l'image aux tokens, puis à la classification • Cliquez sur n'importe quel bloc pour explorer son rôle et ses équations
            </p>
          </div>
        </div>
      </div>

      {/* 2. THREE-COLUMN ARCHITECTURE DIAGRAM */}
      <div className="architecture-three-cols-grid">
        {/* COLONNE 1 : DE L'IMAGE AUX TOKENS */}
        <div className="arch-card-column">
          <div className="arch-col-header-pill">
            <span className="col-num-circle">1</span>
            <span className="col-header-text">De l'image aux tokens</span>
          </div>

          <div className="arch-col-body">
            <div className="image-tokens-pair">
              <div className="img-pair-item">
                <img
                  src={imageUrl}
                  alt="Image 224x224"
                  className="pair-preview-img"
                  onError={(e) => {
                    (e.target as HTMLImageElement).src = '/presets/cat.jpg';
                  }}
                />
                <span className="pair-img-label">
                  Image<br />224 × 224
                </span>
              </div>

              <span className="pair-arrow-right">→</span>

              <div
                className="img-pair-item clickable-block"
                onClick={() => setActiveBlock(BLOCK_DETAILS.patch_embed)}
                title="Cliquer pour inspecter le découpage en patches"
              >
                <div className="gridded-img-wrapper">
                  <img
                    src={imageUrl}
                    alt="Grille 14x14"
                    className="pair-preview-img"
                    onError={(e) => {
                      (e.target as HTMLImageElement).src = '/presets/cat.jpg';
                    }}
                  />
                  <div className="grid-14x14-overlay" />
                </div>
                <span className="pair-img-label">
                  Patch 16 × 16<br />196 patches
                </span>
              </div>
            </div>

            <div className="arch-flow-down-arrow">↓</div>

            {/* Patch Thumbnails Strip */}
            <div
              className="patch-thumbnails-strip clickable-block"
              onClick={() => setActiveBlock(BLOCK_DETAILS.patch_embed)}
              title="Cliquer pour explorer la projection linéaire"
            >
              {Array.from({ length: 6 }).map((_, i) => (
                <div
                  key={i}
                  className="patch-thumb-cell"
                  style={{
                    backgroundImage: `url('${imageUrl}')`,
                    backgroundPosition: `${-(i * 24)}px -20px`,
                    backgroundSize: '160px 120px',
                  }}
                />
              ))}
              <span className="patch-strip-dots">...</span>
              <div
                className="patch-thumb-cell"
                style={{
                  backgroundImage: `url('${imageUrl}')`,
                  backgroundPosition: `-130px -40px`,
                  backgroundSize: '160px 120px',
                }}
              />
            </div>

            {/* Curly Bracket Projection Text */}
            <div
              className="projection-bracket-box clickable-block"
              onClick={() => setActiveBlock(BLOCK_DETAILS.patch_embed)}
            >
              <div className="bracket-curly-line" />
              <div className="bracket-text-group">
                <span className="bracket-title">Projection linéaire</span>
                <span className="bracket-arrow">↓</span>
                <span className="bracket-sub">Vecteurs de dimension 768</span>
              </div>
            </div>

            {/* Sequence of Tokens */}
            <div
              className="tokens-pills-row clickable-block"
              onClick={() => setActiveBlock(BLOCK_DETAILS.pos_embed)}
              title="Cliquer pour inspecter les tokens et l'encodage positionnel"
            >
              <span className="token-node cls-node">[CLS]</span>
              <span className="token-node p-node">P₁</span>
              <span className="token-node p-node">P₂</span>
              <span className="token-node p-node">P₃</span>
              <span className="token-node dots-node">...</span>
              <span className="token-node p-node">P₁₉₆</span>
            </div>

            <button
              type="button"
              className="positional-enc-tag-btn"
              onClick={() => setActiveBlock(BLOCK_DETAILS.pos_embed)}
            >
              + Encodage positionnel 1D
            </button>
          </div>
        </div>

        {/* COLONNE 2 : ARCHITECTURE ViT */}
        <div className="arch-card-column">
          <div className="arch-col-header-pill">
            <span className="col-num-circle">2</span>
            <span className="col-header-text">Architecture ViT</span>
          </div>

          <div className="arch-col-body">
            <div className="arch-node-box light-blue">Image</div>
            <div className="arch-flow-down-arrow">↓</div>

            <div
              className="arch-node-box light-blue clickable-node"
              onClick={() => setActiveBlock(BLOCK_DETAILS.patch_embed)}
            >
              Patch Embedding<br />+ Position Embedding
            </div>
            <div className="arch-flow-down-arrow">↓</div>

            {/* Transformer Encoder Block */}
            <div className="encoder-dashed-container">
              <div className="encoder-repeat-label">
                × L<br />(12<br />blocs)
              </div>

              <div className="encoder-layers-column">
                <div
                  className="arch-node-box light-blue sub-layer clickable-node"
                  onClick={() => setActiveBlock(BLOCK_DETAILS.layernorm)}
                >
                  LayerNorm
                </div>
                <div className="arch-flow-down-arrow mini">↓</div>

                <div
                  className="arch-node-box green-attention sub-layer clickable-node"
                  onClick={() => setActiveBlock(BLOCK_DETAILS.attention)}
                >
                  Multi-Head Self-Attention
                </div>

                <div className="residual-add-circle" title="Connexion résiduelle 1">⊕</div>

                <div
                  className="arch-node-box light-blue sub-layer clickable-node"
                  onClick={() => setActiveBlock(BLOCK_DETAILS.layernorm)}
                >
                  LayerNorm
                </div>
                <div className="arch-flow-down-arrow mini">↓</div>

                <div
                  className="arch-node-box purple-mlp sub-layer clickable-node"
                  onClick={() => setActiveBlock(BLOCK_DETAILS.mlp)}
                >
                  MLP (GELU)
                </div>

                <div className="residual-add-circle" title="Connexion résiduelle 2">⊕</div>

                <svg className="residual-curves-svg" viewBox="0 0 40 240" fill="none">
                  <path d="M 5 20 C 35 20, 35 110, 8 110" stroke="#1e3a8a" strokeWidth="1.5" fill="none" markerEnd="url(#arrowhead)" />
                  <path d="M 5 130 C 35 130, 35 220, 8 220" stroke="#1e3a8a" strokeWidth="1.5" fill="none" markerEnd="url(#arrowhead)" />
                  <defs>
                    <marker id="arrowhead" markerWidth="6" markerHeight="6" refX="4" refY="3" orient="auto">
                      <polygon points="0 0, 6 3, 0 6" fill="#1e3a8a" />
                    </marker>
                  </defs>
                </svg>
              </div>
            </div>

            <div className="arch-flow-down-arrow">↓</div>
            <div
              className="arch-node-box light-blue token-box clickable-node"
              onClick={() => setActiveBlock(BLOCK_DETAILS.cls_token)}
            >
              Token [CLS]
            </div>
            <div className="arch-flow-down-arrow">↓</div>

            <div
              className="arch-node-box light-blue clickable-node"
              onClick={() => setActiveBlock(BLOCK_DETAILS.cls_token)}
            >
              Tête de classification
            </div>
            <div className="arch-flow-down-arrow">↓</div>

            <div className="arch-node-box dark-navy-predicted">Classe prédite (ViT)</div>
          </div>
        </div>

        {/* COLONNE 3 : ARCHITECTURE DeiT */}
        <div className="arch-card-column deit-col">
          <div className="arch-col-header-pill">
            <span className="col-num-circle">3</span>
            <span className="col-header-text">Architecture DeiT</span>
          </div>

          <div className="arch-col-body">
            <div className="arch-node-box light-blue">Image</div>
            <div className="arch-flow-down-arrow">↓</div>

            <div
              className="arch-node-box light-blue clickable-node"
              onClick={() => setActiveBlock(BLOCK_DETAILS.patch_embed)}
            >
              Patch Embedding<br />+ Position Embedding
            </div>
            <div className="arch-flow-down-arrow">↓</div>

            {/* Middle split row: Encoder on Left, Teacher Distillation branch on Right */}
            <div className="deit-middle-split-row">
              {/* Encoder Dashed Block */}
              <div className="encoder-dashed-container deit-encoder-box">
                <div className="encoder-repeat-label">
                  × L<br />(12<br />blocs)
                </div>

                <div className="encoder-layers-column">
                  <div
                    className="arch-node-box light-blue sub-layer clickable-node"
                    onClick={() => setActiveBlock(BLOCK_DETAILS.layernorm)}
                  >
                    LayerNorm
                  </div>
                  <div className="arch-flow-down-arrow mini">↓</div>

                  <div
                    className="arch-node-box green-attention sub-layer clickable-node"
                    onClick={() => setActiveBlock(BLOCK_DETAILS.attention)}
                  >
                    Multi-Head Self-Attention
                  </div>

                  <div className="residual-add-circle" title="Connexion résiduelle 1">⊕</div>

                  <div
                    className="arch-node-box light-blue sub-layer clickable-node"
                    onClick={() => setActiveBlock(BLOCK_DETAILS.layernorm)}
                  >
                    LayerNorm
                  </div>
                  <div className="arch-flow-down-arrow mini">↓</div>

                  <div
                    className="arch-node-box purple-mlp sub-layer clickable-node"
                    onClick={() => setActiveBlock(BLOCK_DETAILS.mlp)}
                  >
                    MLP
                  </div>

                  <div className="residual-add-circle" title="Connexion résiduelle 2">⊕</div>

                  <svg className="residual-curves-svg" viewBox="0 0 40 240" fill="none">
                    <path d="M 5 20 C 35 20, 35 110, 8 110" stroke="#1e3a8a" strokeWidth="1.5" fill="none" markerEnd="url(#arrowhead)" />
                    <path d="M 5 130 C 35 130, 35 220, 8 220" stroke="#1e3a8a" strokeWidth="1.5" fill="none" markerEnd="url(#arrowhead)" />
                  </svg>
                </div>
              </div>

              {/* Teacher Distillation Branch (Visibilité 100% garantie à l'intérieur de la carte) */}
              <div
                className="deit-teacher-branch clickable-block"
                onClick={() => setActiveBlock(BLOCK_DETAILS.dist_token)}
                title="Cliquer pour explorer la distillation CNN"
              >
                <div className="teacher-model-box">
                  Modèle<br />Enseignant<br />(Teacher)
                </div>

                <div className="teacher-dotted-arrow-down">
                  <svg width="24" height="38" viewBox="0 0 24 38">
                    <path d="M 12 0 L 12 30" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="3 3" />
                    <polygon points="7 28, 12 36, 17 28" fill="#8b5cf6" />
                  </svg>
                </div>

                <div className="distillation-pill-box">
                  Distillation
                </div>

                <div className="distillation-curved-arrow-wrap">
                  <svg width="50" height="46" viewBox="0 0 50 46" fill="none">
                    <path d="M 38 2 C 38 28, 14 28, 6 38" stroke="#8b5cf6" strokeWidth="2" strokeDasharray="3 3" fill="none" />
                    <polygon points="1 32, 6 42, 12 35" fill="#8b5cf6" />
                  </svg>
                </div>

                <div className="distillation-desc-callout">
                  DeiT ajoute un mécanisme de distillation pour un apprentissage plus efficace.
                </div>
              </div>
            </div>

            <div className="arch-flow-down-arrow">↓</div>

            {/* Dual Tokens Row (CLS + DIST) */}
            <div className="deit-dual-tokens-row">
              <div
                className="arch-node-box light-blue token-box half clickable-node"
                onClick={() => setActiveBlock(BLOCK_DETAILS.cls_token)}
              >
                [CLS]
              </div>
              <div
                className="arch-node-box purple-token token-box half clickable-node"
                onClick={() => setActiveBlock(BLOCK_DETAILS.dist_token)}
              >
                [DIST]
              </div>
            </div>

            <div className="arch-flow-down-arrow">↓</div>

            {/* Dual Heads */}
            <div className="deit-dual-heads-row">
              <div
                className="arch-node-box green-classification-head half mini clickable-node"
                onClick={() => setActiveBlock(BLOCK_DETAILS.cls_token)}
              >
                Tête<br />(Classification)
              </div>
              <div
                className="arch-node-box purple-distill-head half mini clickable-node"
                onClick={() => setActiveBlock(BLOCK_DETAILS.dist_token)}
              >
                Tête<br />(Distillation)
              </div>
            </div>

            <div className="arch-flow-down-arrow">↓</div>

            <div className="arch-node-box dark-navy-predicted">
              Prédiction finale
            </div>

            <div className="deit-footer-callout">
              <span className="deit-footer-icon">ℹ</span>
              <span>Le token <strong>[DIST]</strong> apprend à partir des sorties de l'enseignant pendant l'entraînement.</span>
            </div>
          </div>
        </div>
      </div>

      {/* 2.5 BANNIÈRES DE SYNTHÈSE (Directement sous les 3 colonnes) */}
      <div className="arch-bottom-summary-banners">
        {/* Barre bleue foncée */}
        <div className="arch-navy-summary-banner">
          <div className="navy-banner-side">
            <span className="navy-banner-model">ViT :</span> prédiction basée sur <span className="navy-token-tag">[CLS]</span>
          </div>
          <div className="navy-banner-divider" />
          <div className="navy-banner-side">
            <span className="navy-banner-model">DeiT :</span> prédiction + token <span className="navy-token-tag purple">[DIST]</span> guidé pendant l'entraînement par un enseignant
          </div>
        </div>

        {/* Barre dorée / orange */}
        <div className="arch-orange-insight-banner">
          <span className="orange-insight-icon">🎯</span>
          <span className="orange-insight-text">
            <strong>Les deux modèles transforment l'image en séquence de tokens</strong> ; DeiT enrichit l'apprentissage de ViT par la distillation.
          </span>
        </div>
      </div>

      {/* 3. MODAL DRAWER DE DÉTAIL DU BLOC SÉLECTIONNÉ */}
      {activeBlock && (
        <div className="modal-backdrop" onClick={() => setActiveBlock(null)}>
          <div className="block-detail-drawer" onClick={(e) => e.stopPropagation()}>
            <div className="drawer-header">
              <div className="drawer-header-left">
                <span className="drawer-tag">COMPOSANT ARCHITECTURAL</span>
                <h3>{activeBlock.title}</h3>
              </div>
              <button
                type="button"
                className="drawer-close-btn"
                onClick={() => setActiveBlock(null)}
              >
                <X size={20} />
              </button>
            </div>

            <div className="drawer-body">
              <div className="drawer-formula-card">
                <span className="formula-card-label">Formulation Mathématique (KaTeX) :</span>
                <BlockMath math={activeBlock.formula} />
              </div>

              <div className="drawer-text-block">
                <h4>Rôle dans l'Architecture :</h4>
                <p>{activeBlock.role}</p>
              </div>

              <div className="drawer-text-block">
                <h4>Pourquoi cette étape ? (Intuition Deep Learning) :</h4>
                <p>{activeBlock.why}</p>
              </div>

              <div className="drawer-actions-row">
                <button
                  type="button"
                  className="btn-drawer-action primary"
                  onClick={() => {
                    navigate('/calculs-manuels');
                  }}
                >
                  <Calculator size={16} />
                  <span>Calculer manuellement cette étape</span>
                </button>
                <button
                  type="button"
                  className="btn-drawer-action secondary"
                  onClick={() => {
                    navigate('/pratiquer');
                  }}
                >
                  <Play size={16} />
                  <span>Voir le tenseur réel dans Pratiquer</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 4. TABLEAU COMPARATIF DÉTAILLÉ VIT VS DEIT */}
      <section className="unified-comparison-section">
        <div className="section-title-wrap">
          <span className="section-kicker">SYNTHÈSE TECHNIQUE</span>
          <h2>Tableau comparatif approfondi ViT vs DeiT</h2>
        </div>

        <div className="table-responsive-wrapper">
          <table className="architecture-comparison-table">
            <thead>
              <tr>
                <th>Critère</th>
                <th>Google ViT-Base (2020)</th>
                <th>Meta DeiT-Base Distilled (2021)</th>
                <th>Impact &amp; Différence Clé</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><strong>Données d'entraînement</strong></td>
                <td>JFT-300M (propriétaire) puis fine-tuning ImageNet</td>
                <td>ImageNet-1k seul (1.2M images)</td>
                <td>DeiT élimine la dépendance aux bases de données privées colossales</td>
              </tr>
              <tr>
                <td><strong>Mécanisme de distillation</strong></td>
                <td>Aucun (supervisé classique)</td>
                <td>Distillation par attention avec token <code>[DIST]</code></td>
                <td>Le token spécial apprend directement des activations du professeur CNN</td>
              </tr>
              <tr>
                <td><strong>Modèle Enseignant (Teacher)</strong></td>
                <td>N/A</td>
                <td>RegNetY-160 (Réseau de neurones convolutif)</td>
                <td>Transfère le biais inductif spatial (localité) au Transformer</td>
              </tr>
              <tr>
                <td><strong>Nombre de tokens</strong></td>
                <td>197 tokens (196 patches + 1 [CLS])</td>
                <td>198 tokens (196 patches + 1 [CLS] + 1 [DIST])</td>
                <td>Un token additionnel dédié à la communication avec le teacher</td>
              </tr>
              <tr>
                <td><strong>Inférence finale</strong></td>
                <td>Logits de la tête <code>[CLS]</code></td>
                <td>Moyenne des probabilités : <InlineMath math="\frac{1}{2}(P_{\text{cls}} + P_{\text{dist}})" /></td>
                <td>L'accord entre les deux têtes renforce la robustesse de prédiction</td>
              </tr>
              <tr>
                <td><strong>Augmentations de données</strong></td>
                <td>Standard (RandAugment, mixup)</td>
                <td>Fortes : RandAugment, Mixup, CutMix, Repeated Augmentation</td>
                <td>Essentielles pour éviter le surapprentissage sans grand jeu de données</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      {/* 5. QUIZ DE COMPRÉHENSION INTERACTIF */}
      <section className="comprehension-quiz-section">
        <div className="section-title-wrap">
          <span className="section-kicker">VALIDATION DES ACQUIS</span>
          <h2>Quiz de Compréhension Architecturale</h2>
          <p className="quiz-subtitle">
            Testez vos connaissances sur ViT et DeiT pour gagner des points XP et débloquer vos badges.
          </p>
        </div>

        <div className="quiz-cards-list">
          {quizQuestions.map((q) => {
            const isAnswered = quizAnswers[q.id] !== undefined;
            const isCorrect = quizAnswers[q.id] === q.correct;

            return (
              <div key={q.id} className="quiz-card-item">
                <span className="quiz-card-q">Question {q.id} : {q.q}</span>

                <div className="quiz-options-list">
                  {q.options.map((opt, oIdx) => {
                    const isSelected = quizAnswers[q.id] === oIdx;
                    let optClass = 'quiz-option-btn';
                    if (isSelected) optClass += ' selected';
                    if (quizSubmitted) {
                      if (oIdx === q.correct) optClass += ' correct';
                      else if (isSelected && !isCorrect) optClass += ' incorrect';
                    }

                    return (
                      <button
                        key={oIdx}
                        type="button"
                        className={optClass}
                        onClick={() => handleSelectAnswer(q.id, oIdx)}
                      >
                        <span className="opt-letter">
                          {String.fromCharCode(65 + oIdx)}.
                        </span>
                        <span className="opt-text">{opt}</span>
                        {quizSubmitted && oIdx === q.correct && (
                          <CheckCircle2 size={16} className="text-emerald-500 ml-auto" />
                        )}
                        {quizSubmitted && isSelected && !isCorrect && (
                          <XCircle size={16} className="text-red-500 ml-auto" />
                        )}
                      </button>
                    );
                  })}
                </div>

                {quizSubmitted && (
                  <div className={`quiz-feedback-box ${isCorrect ? 'correct' : 'incorrect'}`}>
                    <strong>{isCorrect ? 'Excellente réponse !' : 'À revoir :'}</strong> {q.explanation}
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="quiz-submit-row">
          {!quizSubmitted ? (
            <button
              type="button"
              className="btn-submit-quiz"
              onClick={submitQuiz}
              disabled={Object.keys(quizAnswers).length < quizQuestions.length}
            >
              <Award size={16} />
              <span>Valider mes réponses au Quiz (+10 XP)</span>
            </button>
          ) : (
            <button
              type="button"
              className="btn-submit-quiz secondary"
              onClick={() => {
                setQuizAnswers({});
                setQuizSubmitted(false);
              }}
            >
              <span>Recommencer le Quiz</span>
            </button>
          )}
        </div>
      </section>
    </div>
  );
}
