import React, { useState } from 'react';
import {
  Calculator,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  XCircle,
  HelpCircle,
  RotateCcw,
  BookOpen,
  Eye,
  Sliders,
  Layers,
  Zap,
  Sparkles,
  Grid,
  Box,
  TrendingUp,
  Award,
  Check,
  Cpu,
  Brain,
  MousePointer2,
  Info,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { InlineMath, BlockMath, TeX } from '../../components/MathFormula';
import {
  calculatePatchStats,
  isClose,
  calculateLayerNorm,
  geluExact,
  calculateSoftmax,
} from './lab/mathEngine';
import { PatchSizeOption, ValidationFeedback } from './lab/types';

// Interactive Embedding Ribbon Component
function EmbeddingRibbon({ patchSize }: { patchSize: number }) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const totalPatches = Math.floor(224 / patchSize) ** 2;
  const dim = patchSize * patchSize * 3;

  // Generate a pseudo-random color gradient for each embedding based on index
  const getBarColor = (i: number) => {
    const hue = (i * 37 + 200) % 360;
    const isHovered = hoveredIdx === i;
    const isSelected = selectedIdx === i;
    if (isSelected) return `hsl(${hue}, 90%, 45%)`;
    if (isHovered) return `hsl(${hue}, 80%, 55%)`;
    return `hsl(${hue}, 60%, 65%)`;
  };

  // Show up to 20 bars + "..." + last bar
  const visibleBars = Math.min(20, totalPatches);
  const showDots = totalPatches > 21;

  return (
    <div className="embed-ribbon-container">
      <div className="embed-ribbon-header">
        <span className="embed-ribbon-title">
          Ruban de <strong>{totalPatches}</strong> vecteurs latents
          <span className="embed-ribbon-dim-badge">dim = {dim}</span>
        </span>
        <span className="embed-ribbon-hint">
          <svg width="13" height="13" viewBox="0 0 13 13" fill="none" style={{display:'inline',marginRight:3}}>
            <circle cx="6.5" cy="6.5" r="6" stroke="#6366f1" strokeWidth="1.2"/>
            <text x="6.5" y="10" textAnchor="middle" fill="#6366f1" fontSize="8" fontWeight="bold">i</text>
          </svg>
          Survolez ou cliquez un patch
        </span>
      </div>

      <div className="embed-ribbon-scroll">
        {Array.from({ length: visibleBars }).map((_, i) => (
          <div
            key={i}
            className={`embed-vbar ${selectedIdx === i ? 'selected' : ''} ${hoveredIdx === i ? 'hovered' : ''}`}
            style={{ background: getBarColor(i) }}
            onMouseEnter={() => setHoveredIdx(i)}
            onMouseLeave={() => setHoveredIdx(null)}
            onClick={() => setSelectedIdx(selectedIdx === i ? null : i)}
            title={`Patch #${i + 1} → vecteur ℝ^${dim}`}
          />
        ))}
        {showDots && (
          <span className="embed-vbar-dots">···</span>
        )}
        {totalPatches > visibleBars && (
          <div
            key="last"
            className={`embed-vbar ${selectedIdx === totalPatches - 1 ? 'selected' : ''}`}
            style={{ background: getBarColor(totalPatches - 1) }}
            onMouseEnter={() => setHoveredIdx(totalPatches - 1)}
            onMouseLeave={() => setHoveredIdx(null)}
            onClick={() => setSelectedIdx(selectedIdx === totalPatches - 1 ? null : totalPatches - 1)}
            title={`Patch #${totalPatches} → vecteur ℝ^${dim}`}
          />
        )}
      </div>

      {/* Detail card for selected/hovered patch */}
      {(hoveredIdx !== null || selectedIdx !== null) && (
        <div className="embed-detail-card">
          {(() => {
            const idx = hoveredIdx ?? selectedIdx!;
            const row = Math.floor(idx / Math.floor(224 / patchSize)) + 1;
            const col = (idx % Math.floor(224 / patchSize)) + 1;
            return (
              <>
                <div className="embed-detail-header">
                  <span className="embed-detail-pill">Patch #{idx + 1}</span>
                  <span className="embed-detail-pos">Ligne {row}, Colonne {col}</span>
                </div>
                <div className="embed-detail-body">
                  <div className="embed-detail-row">
                    <span className="embed-detail-label">Entrée (aplatie)</span>
                    <span className="embed-detail-val">
                      <InlineMath math={`\\mathbf{x}_p \\in \\mathbb{R}^{${dim}}`} />
                    </span>
                  </div>
                  <div className="embed-detail-arrow">
                    <svg width="80" height="18" viewBox="0 0 80 18">
                      <line x1="4" y1="9" x2="68" y2="9" stroke="#6366f1" strokeWidth="2" markerEnd="url(#arr)" />
                      <defs>
                        <marker id="arr" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
                          <polygon points="0 0, 6 3, 0 6" fill="#6366f1" />
                        </marker>
                      </defs>
                      <text x="22" y="7" fontSize="8" fill="#6366f1" fontWeight="600">E × xₚ + b</text>
                    </svg>
                  </div>
                  <div className="embed-detail-row">
                    <span className="embed-detail-label">Embedding (sortie)</span>
                    <span className="embed-detail-val">
                      <InlineMath math={`\\mathbf{z}_{${idx+1}} \\in \\mathbb{R}^{768}`} />
                    </span>
                  </div>
                  <div className="embed-minibar-vis">
                    {Array.from({ length: 16 }).map((_, j) => {
                      const v = Math.abs(Math.sin((idx + 1) * 7.3 + j * 2.1));
                      return (
                        <div
                          key={j}
                          className="embed-minibar-cell"
                          style={{ height: `${12 + v * 24}px`, opacity: 0.5 + v * 0.5 }}
                          title={`dim ${j + 1}: ${v.toFixed(3)}`}
                        />
                      );
                    })}
                    <span className="embed-minibar-dots">···768</span>
                  </div>
                </div>
              </>
            );
          })()}
        </div>
      )}

      {/* Summary stats */}
      <div className="embed-ribbon-stats">
        <div className="embed-stat-box">
          <span className="embed-stat-num">{totalPatches}</span>
          <span className="embed-stat-lbl">patches</span>
        </div>
        <div className="embed-stat-box">
          <span className="embed-stat-num">×</span>
          <span className="embed-stat-lbl" />
        </div>
        <div className="embed-stat-box">
          <span className="embed-stat-num">{dim}</span>
          <span className="embed-stat-lbl">dim. entrée</span>
        </div>
        <div className="embed-stat-box">
          <span className="embed-stat-num">→</span>
          <span className="embed-stat-lbl" />
        </div>
        <div className="embed-stat-box blue">
          <span className="embed-stat-num">{totalPatches}</span>
          <span className="embed-stat-lbl">vecteurs</span>
        </div>
        <div className="embed-stat-box">
          <span className="embed-stat-num">×</span>
          <span className="embed-stat-lbl" />
        </div>
        <div className="embed-stat-box blue">
          <span className="embed-stat-num">768</span>
          <span className="embed-stat-lbl">dim. sortie</span>
        </div>
      </div>
    </div>
  );
}

// Interactive Token Sequence Visualizer for Step 3
function TokenSequenceVisualizer({ model, patchSize }: { model: 'vit' | 'deit'; patchSize: number }) {
  const [hoveredTok, setHoveredTok] = useState<string | null>(null);
  const totalPatches = Math.floor(224 / patchSize) ** 2;
  const seqLen = model === 'deit' ? totalPatches + 2 : totalPatches + 1;

  const tokenDefs = [
    { id: 'cls', label: '[CLS]', color: '#1e40af', bg: '#eff6ff', border: '#93c5fd',
      desc: 'Token de classification. Son embedding final est utilisé pour la prédiction de classe par la tête MLP.' },
    ...(model === 'deit' ? [{
      id: 'dist', label: '[DIST]', color: '#6d28d9', bg: '#f5f3ff', border: '#c4b5fd',
      desc: 'Token de distillation (DeiT uniquement). Il guide l\'apprentissage en imitant les sorties du modèle enseignant CNN.'
    }] : []),
    { id: 'p1', label: 'P₁', color: '#065f46', bg: '#ecfdf5', border: '#6ee7b7',
      desc: `Premier patch embedding. Représentation vectorielle (ℝ^768) du patch en haut à gauche de l'image.` },
    { id: 'p2', label: 'P₂', color: '#065f46', bg: '#ecfdf5', border: '#6ee7b7',
      desc: 'Deuxième patch embedding (patch suivant dans la séquence de gauche à droite, haut en bas).' },
    { id: 'dots', label: '···', color: '#64748b', bg: '#f8fafc', border: '#e2e8f0',
      desc: `${totalPatches - 2} autres patch embeddings (P₃ à P${totalPatches - 1})` },
    { id: `p${totalPatches}`, label: `P${totalPatches}`, color: '#065f46', bg: '#ecfdf5', border: '#6ee7b7',
      desc: `Dernier patch embedding (patch en bas à droite). Total : ${totalPatches} patches.` },
  ];

  return (
    <div className="tokseq-wrapper">
      <div className="tokseq-header">
        <span className="tokseq-title">
          Séquence <strong>{model === 'vit' ? 'ViT' : 'DeiT'}</strong> :
          <span className="tokseq-len-badge">{seqLen} tokens</span>
        </span>
        <span className="tokseq-hint">Survolez un token pour explorer son rôle</span>
      </div>

      <div className="tokseq-row">
        {tokenDefs.map((tok) => (
          <div
            key={tok.id}
            className={`tokseq-pill ${hoveredTok === tok.id ? 'active' : ''}`}
            style={{ color: tok.color, background: tok.bg, borderColor: tok.border }}
            onMouseEnter={() => setHoveredTok(tok.id)}
            onMouseLeave={() => setHoveredTok(null)}
          >
            {tok.label}
          </div>
        ))}
        <span className="tokseq-formula">
          + <InlineMath math="\mathbf{E}_{\text{pos}}" />
        </span>
      </div>

      {hoveredTok && (() => {
        const tok = tokenDefs.find(t => t.id === hoveredTok);
        if (!tok) return null;
        return (
          <div className="tokseq-tooltip" style={{ borderColor: tok.border }}>
            <span className="tokseq-tooltip-tag" style={{ background: tok.bg, color: tok.color, borderColor: tok.border }}>
              {tok.label}
            </span>
            <p className="tokseq-tooltip-text">{tok.desc}</p>
          </div>
        );
      })()}

      <div className="tokseq-pos-add">
        <div className="tokseq-pos-row">
          <span className="tokseq-pos-label">Embedding du token :</span>
          <span className="tokseq-vec-example">
            <InlineMath math="\mathbf{E} = [0.5,\; -0.2,\; 1.0]" />
          </span>
        </div>
        <div className="tokseq-pos-row">
          <span className="tokseq-pos-label">Encodage positionnel :</span>
          <span className="tokseq-vec-example">
            <InlineMath math="\mathbf{E}_{\text{pos}} = [0.1,\; 0.4,\; -0.3]" />
          </span>
        </div>
        <div className="tokseq-pos-row result">
          <span className="tokseq-pos-label">Résultat <InlineMath math="\mathbf{Z}" /> :</span>
          <span className="tokseq-vec-example green">
            <InlineMath math="\mathbf{Z} = [0.6,\; 0.2,\; 0.7]" />
          </span>
        </div>
      </div>
    </div>
  );
}

export function TheoreticalCalculations() {
  const {
    imageUrl,
    selectedModel,
    setSelectedModel,
    diffLevel,
    setDiffLevel,
    score,
    attempts,
    validatedTheorySteps,
    validateTheoryStep,
    registerAttempt,
    resetAllProgress,
  } = useAppStore();

  const [currentStep, setCurrentStep] = useState<number>(1);
  const [hintLevels, setHintLevels] = useState<Record<number, number>>({});
  const [solutionVisible, setSolutionVisible] = useState<Record<number, boolean>>({});

  // -------------------------------------------------------------
  // STEP 1 STATE: Découpage
  // -------------------------------------------------------------
  const [step1PatchSize, setStep1PatchSize] = useState<PatchSizeOption>(16);
  const [step1Inputs, setStep1Inputs] = useState({
    gridRowsCols: '',
    totalPatches: '',
    flattenedDim: '',
  });
  const [step1Feedback, setStep1Feedback] = useState<ValidationFeedback | null>(null);

  // -------------------------------------------------------------
  // STEP 2 STATE: Projection Linéaire & Matrix Mult
  // -------------------------------------------------------------
  const [step2Inputs, setStep2Inputs] = useState({
    dimIn: '',
    dimOut: '',
    totalParams: '',
    z1: '',
    z2: '',
  });
  const [step2Feedback, setStep2Feedback] = useState<ValidationFeedback | null>(null);

  // -------------------------------------------------------------
  // STEP 3 STATE: Tokens & Position
  // -------------------------------------------------------------
  const [step3Inputs, setStep3Inputs] = useState({
    seqLenViT: '',
    seqLenDeiT: '',
    tensorShapeBatch1: '',
    zPos1: '',
    zPos2: '',
    zPos3: '',
  });
  const [step3Feedback, setStep3Feedback] = useState<ValidationFeedback | null>(null);

  // -------------------------------------------------------------
  // STEP 4 STATE: Layer Normalization
  // -------------------------------------------------------------
  const [step4Inputs, setStep4Inputs] = useState({
    mean: '',
    variance: '',
    stdDev: '',
    xNorm1: '',
    xNorm2: '',
    xNorm3: '',
    xNorm4: '',
  });
  const [step4Feedback, setStep4Feedback] = useState<ValidationFeedback | null>(null);

  // -------------------------------------------------------------
  // STEP 5 STATE: Multi-Head Self-Attention
  // -------------------------------------------------------------
  const [step5Inputs, setStep5Inputs] = useState({
    kt11: '',
    kt12: '',
    kt21: '',
    kt22: '',
    qkt11: '',
    qkt12: '',
    s11: '',
    s12: '',
    a11: '',
    a12: '',
  });
  const [step5Feedback, setStep5Feedback] = useState<ValidationFeedback | null>(null);

  // -------------------------------------------------------------
  // STEP 6 STATE: MLP & GELU
  // -------------------------------------------------------------
  const [step6Inputs, setStep6Inputs] = useState({
    z1: '',
    geluZ1: '',
    geluZ2: '',
    expansionFactor: '',
    mlpDim: '',
  });
  const [step6Feedback, setStep6Feedback] = useState<ValidationFeedback | null>(null);

  // -------------------------------------------------------------
  // STEP 7 STATE: Classification & Distillation
  // -------------------------------------------------------------
  const [step7Inputs, setStep7Inputs] = useState({
    exp1: '',
    exp2: '',
    exp3: '',
    sumExp: '',
    prob1: '',
    winningClass: '',
    pFinal1: '',
    pFinal2: '',
    pFinal3: '',
  });
  const [step7Feedback, setStep7Feedback] = useState<ValidationFeedback | null>(null);

  const stepsList = [
    { id: 1, title: 'Découpage en patches' },
    { id: 2, title: 'Projection linéaire' },
    { id: 3, title: 'Tokens & positions' },
    { id: 4, title: 'Layer Normalization' },
    { id: 5, title: 'Attention (Q, K, V)' },
    { id: 6, title: 'MLP & GELU' },
    { id: 7, title: 'Classification & Distillation' },
  ];

  // Helper validation
  const recordValidation = (stepId: number, isSuccess: boolean) => {
    registerAttempt();
    if (isSuccess) {
      validateTheoryStep(stepId, 15);
    }
  };

  const showNextHint = (stepId: number) => {
    setHintLevels((prev) => ({
      ...prev,
      [stepId]: Math.min(3, (prev[stepId] || 0) + 1),
    }));
  };

  // -------------------------------------------------------------
  // VERIFICATIONS
  // -------------------------------------------------------------
  const verifyStep1 = () => {
    // Calcul selon la taille de patch sélectionnée par l'utilisateur (image 224x224)
    const H_real = 224;
    const W_real = 224;
    const P_real = step1PatchSize; // 8, 16 ou 32
    const C = 3;
    const statsReal = calculatePatchStats(H_real, W_real, P_real, C);

    // Statistiques miniatures alternatives (au cas où l'utilisateur a résolu sur 6x6 avec P=2)
    const statsMini = calculatePatchStats(6, 6, 2, C);

    const okGridReal = isClose(step1Inputs.gridRowsCols, statsReal.gridRows, 0.1);
    const okTotalReal = isClose(step1Inputs.totalPatches, statsReal.totalPatches, 0.1);
    const okDimReal = isClose(step1Inputs.flattenedDim, statsReal.flattenedDim, 0.1);

    const okGridMini = isClose(step1Inputs.gridRowsCols, statsMini.gridRows, 0.1);
    const okTotalMini = isClose(step1Inputs.totalPatches, statsMini.totalPatches, 0.1);
    const okDimMini = isClose(step1Inputs.flattenedDim, statsMini.flattenedDim, 0.1);

    const isSuccess = (okGridReal && okTotalReal && okDimReal) || (okGridMini && okTotalMini && okDimMini);
    recordValidation(1, isSuccess);

    setStep1Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? `Parfait ! Pour P = ${step1PatchSize}×${step1PatchSize}, vous obtenez bien ${statsReal.gridRows} patches par côté, N = ${statsReal.totalPatches} patches au total et D = ${statsReal.flattenedDim} valeurs par vecteur.`
        : `Pour une image 224×224 avec P = ${step1PatchSize}×${step1PatchSize} : calculez H/P = 224/${step1PatchSize} = ${statsReal.gridRows}, N = (${statsReal.gridRows})² = ${statsReal.totalPatches}, et D = ${step1PatchSize}²×3 = ${statsReal.flattenedDim}.`,
    });
  };

  const verifyStep2 = () => {
    // Accepter aussi bien la dimension réelle ViT (768, 768) que miniature (12, 4) ou basée sur le patch précédent
    const okDin =
      isClose(step2Inputs.dimIn, 768, 0.1) ||
      isClose(step2Inputs.dimIn, 12, 0.1) ||
      isClose(step2Inputs.dimIn, step1PatchSize * step1PatchSize * 3, 0.1);

    const okDout =
      isClose(step2Inputs.dimOut, 768, 0.1) ||
      isClose(step2Inputs.dimOut, 4, 0.1);

    // Mini test: x = [2, 1, 3], W = [[1, 0], [0, 2], [1, 1]], b = [1, -1]
    // z = [2*1 + 1*0 + 3*1 + 1, 2*0 + 1*2 + 3*1 - 1] = [6, 4]
    const okZ1 = isClose(step2Inputs.z1, 6, 0.1);
    const okZ2 = isClose(step2Inputs.z2, 4, 0.1);

    const isSuccess = okDin && okDout && okZ1 && okZ2;
    recordValidation(2, isSuccess);

    setStep2Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? "Bravo ! La projection linéaire z = xW + b transforme chaque patch aplati en un vecteur d'embedding."
        : 'Vérifiez z₁ = 2(1) + 1(0) + 3(1) + 1 = 6, et z₂ = 2(0) + 1(2) + 3(1) - 1 = 4. D_in = 768 (ou 12 en miniature), D_model = 768 (ou 4 en miniature).',
    });
  };

  const verifyStep3 = () => {
    const isVit = selectedModel === 'vit';
    const numPatches = diffLevel === 'miniature' ? 9 : 196;

    // ViT attend N + 1 tokens (197 réels ou 10 miniatures)
    const okViT = isClose(step3Inputs.seqLenViT, numPatches + 1, 0.1) || isClose(step3Inputs.seqLenViT, 197, 0.1) || isClose(step3Inputs.seqLenViT, 10, 0.1);
    // DeiT attend N + 2 tokens (198 réels ou 11 miniatures avec [DIST])
    const okDeiT = isClose(step3Inputs.seqLenDeiT, numPatches + 2, 0.1) || isClose(step3Inputs.seqLenDeiT, 198, 0.1) || isClose(step3Inputs.seqLenDeiT, 11, 0.1);

    // Vector addition: E = [0.5, -0.2, 1.0], E_pos = [0.1, 0.4, -0.3] -> Z = [0.6, 0.2, 0.7]
    const okZ1 = isClose(step3Inputs.zPos1, 0.6, 0.08);
    const okZ2 = isClose(step3Inputs.zPos2, 0.2, 0.08);
    const okZ3 = isClose(step3Inputs.zPos3, 0.7, 0.08);
    const okPosAddition = okZ1 && okZ2 && okZ3;

    // La validation s'adapte au modèle actif sélectionné !
    const okActiveLength = isVit
      ? (okViT || (step3Inputs.seqLenViT === '' && okDeiT))
      : (okDeiT || (step3Inputs.seqLenDeiT === '' && okViT));

    const isSuccess = okActiveLength && okPosAddition;
    recordValidation(3, isSuccess);

    setStep3Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? (isVit
            ? "Parfait ! En mode ViT, la séquence compte exactement N + 1 = 197 tokens (aucun token [DIST]). L'addition positionnelle Z = E + E_pos est correcte."
            : "Bravo ! En mode DeiT, la séquence compte N + 2 = 198 tokens grâce à l'insertion du token [DIST]. L'addition positionnelle Z = E + E_pos est exacte.")
        : (isVit
            ? `Pour ViT (${numPatches} patches) : Longueur = ${numPatches} + 1 ([CLS]) = ${numPatches + 1} tokens. Z = [0.5+0.1, -0.2+0.4, 1.0-0.3] = [0.6, 0.2, 0.7].`
            : `Pour DeiT (${numPatches} patches) : Longueur = ${numPatches} + 2 ([CLS] + [DIST]) = ${numPatches + 2} tokens. Z = [0.6, 0.2, 0.7].`),
    });
  };

  const verifyStep4 = () => {
    // Vector: [1.0, 3.0, 3.0, 9.0] -> mean = 4.0, var = 9.0, std = 3.0
    // normalized = [-1.0, -0.333, -0.333, 1.667]
    const okMean = isClose(step4Inputs.mean, 4.0, 0.15);
    const okVar = isClose(step4Inputs.variance, 9.0, 0.25);
    const okStd = isClose(step4Inputs.stdDev, 3.0, 0.15);
    const okN1 = isClose(step4Inputs.xNorm1, -1.0, 0.15);
    const okN2 = isClose(step4Inputs.xNorm2, -0.333, 0.15);
    const okN3 = isClose(step4Inputs.xNorm3, -0.333, 0.15);
    const okN4 = isClose(step4Inputs.xNorm4, 1.667, 0.15);

    const isSuccess = okMean && okVar && okStd && okN1 && okN2 && okN3 && okN4;
    recordValidation(4, isSuccess);

    setStep4Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? 'Superbe ! Layer Normalization stabilise la variance et centre les activations sur chaque token.'
        : 'Rappel : μ = (1+3+3+9)/4 = 4.0, σ² = ((1-4)² + (3-4)² + (3-4)² + (9-4)²)/4 = (9+1+1+25)/4 = 36/4 = 9.0, σ = 3.0. Normalisés : [-1.0, -0.33, -0.33, 1.67].',
    });
  };

  const verifyStep5 = () => {
    // Q = [[2, 0], [1, 1]], K = [[1, 1], [0, 2]]
    // KT = [[1, 0], [1, 2]]
    // QKT = [[2, 0], [2, 2]]
    // Scale by sqrt(4) = 2 -> S11 = 1.0, S12 = 0.0
    // Softmax row 1: e^1 / (e^1 + e^0) = 2.718 / 3.718 ≈ 0.731
    const okKt =
      isClose(step5Inputs.kt11, 1, 0.15) &&
      isClose(step5Inputs.kt12, 0, 0.15) &&
      isClose(step5Inputs.kt21, 1, 0.15) &&
      isClose(step5Inputs.kt22, 2, 0.15);

    const okQkt = isClose(step5Inputs.qkt11, 2, 0.15) && isClose(step5Inputs.qkt12, 0, 0.15);
    const okS = isClose(step5Inputs.s11, 1.0, 0.15) && isClose(step5Inputs.s12, 0.0, 0.15);
    const okA = isClose(step5Inputs.a11, 0.73, 0.1) && isClose(step5Inputs.a12, 0.27, 0.1);

    const isSuccess = okKt && okQkt && okS && okA;
    recordValidation(5, isSuccess);

    setStep5Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? "Impressionnant ! Vous avez calculé le produit scalaire QKᵀ, l'échelle √d_k et le Softmax ligne par ligne."
        : 'Attention : Kᵀ = [[1, 0], [1, 2]], QKᵀ = [[2, 0], [2, 2]], échelle (/2) = [[1, 0], ...], Softmax : a₁₁ ≈ 0.73, a₁₂ ≈ 0.27.',
    });
  };

  const verifyStep6 = () => {
    // x = [2, -1], W1 = [[1], [-1]], b = 0 -> z1 = 2(1) + (-1)(-1) = 3.0
    // GELU(3.0) ≈ 2.996
    // GELU(-1.0) ≈ -0.159
    const okZ1 = isClose(step6Inputs.z1, 3.0, 0.15);
    const okGelu1 = isClose(step6Inputs.geluZ1, 2.996, 0.15) || isClose(step6Inputs.geluZ1, 3.0, 0.1);
    const okGelu2 = isClose(step6Inputs.geluZ2, -0.159, 0.1) || isClose(step6Inputs.geluZ2, -0.16, 0.08);
    const okExp = isClose(step6Inputs.expansionFactor, 4, 0.15);
    const okMlp = isClose(step6Inputs.mlpDim, 3072, 10) || isClose(step6Inputs.mlpDim, 16, 2);

    const isSuccess = okZ1 && okGelu1 && okGelu2 && okExp && okMlp;
    recordValidation(6, isSuccess);

    setStep6Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? "Bravo ! Vous comprenez l'activation continue GELU et l'expansion dimensionnelle 4×D = 3072."
        : 'Vérifiez z₁ = 2(1) + (-1)(-1) = 3.0. GELU(3.0) ≈ 2.996, GELU(-1.0) ≈ -0.159. Facteur = 4, D_MLP = 4×768 = 3072.',
    });
  };

  const verifyStep7 = () => {
    const isDeit = selectedModel === 'deit';
    // Logits: [2.0, 1.0, 0.0] -> sum exp = 7.389 + 2.718 + 1.0 = 11.107 -> prob1 = 0.665
    const okExp1 = isClose(step7Inputs.exp1, 7.389, 0.2);
    const okSum = isClose(step7Inputs.sumExp, 11.107, 0.35);
    const okP1 = isClose(step7Inputs.prob1, 0.665, 0.08);
    const okWin = isClose(step7Inputs.winningClass, 1, 0.1) || isClose(step7Inputs.winningClass, 0, 0.1);

    // Pour DeiT : fusion des 2 têtes 0.5 * (P_cls + P_dist) -> [0.70, 0.225, 0.075]
    const okPf1 = isClose(step7Inputs.pFinal1, 0.7, 0.08);
    const okPf2 = isClose(step7Inputs.pFinal2, 0.225, 0.08);

    const isSuccess = isDeit
      ? (okExp1 && okSum && okP1 && okPf1 && okPf2)
      : (okExp1 && okSum && okP1);

    recordValidation(7, isSuccess);

    setStep7Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? (isDeit
            ? "Félicitations ! Pour DeiT, les probabilités des têtes [CLS] et [DIST] sont fusionnées : P_final₁ = (0.60 + 0.80)/2 = 0.70, P_final₂ = 0.225. L'apprentissage guidé par l'enseignant CNN est validé !"
            : "Bravo ! Pour ViT, la classification finale repose uniquement sur le vecteur [CLS] : P(y=1) ≈ 66.5% (classe prédite = 1). Aucun mécanisme de distillation n'intervient dans l'architecture originale.")
        : (isDeit
            ? "Pour DeiT : calculez exp(2.0) ≈ 7.389, Somme ≈ 11.107, P₁ ≈ 0.665, puis la moyenne des deux têtes : P_final₁ = (0.60 + 0.80)/2 = 0.70."
            : "Pour ViT : calculez exp(2.0) ≈ 7.389, Somme ≈ 11.107, et la probabilité Softmax P(y=1) = 7.389 / 11.107 ≈ 0.665 (66.5%)."),
    });
  };

  const completedCount = Object.keys(validatedTheorySteps).length;
  const progressPct = Math.round((completedCount / 7) * 100);

  return (
    <div className="unified-page-container">
      {/* 1. HERO BANNER */}
      <div className="unified-banner-card">
        <div className="banner-left-content">
          <div className="banner-icon-box">
            <Calculator size={28} className="banner-main-icon" />
          </div>
          <div className="banner-titles">
            <h1 className="banner-title">Laboratoire théorique : de l'image à la classification</h1>
            <p className="banner-subtitle">
              Réalisez vous-même les calculs mathématiques d'un Vision Transformer et d'un DeiT • Formules KaTeX &amp; tolérance numérique
            </p>
          </div>
        </div>

        {/* Global Selectors: Model & Level */}
        <div className="banner-controls-group">
          <div className="control-pill-toggle">
            <span className="pill-toggle-label">Architecture active :</span>
            <button
              type="button"
              className={`pill-toggle-btn vit-btn ${selectedModel === 'vit' ? 'active' : ''}`}
              onClick={() => setSelectedModel('vit')}
              title="Architecture Google ViT standard"
            >
              <Brain size={15} />
              <span>ViT</span>
              {selectedModel === 'vit' && <span className="active-dot vit" />}
            </button>
            <button
              type="button"
              className={`pill-toggle-btn deit-btn ${selectedModel === 'deit' ? 'active' : ''}`}
              onClick={() => setSelectedModel('deit')}
              title="Architecture Meta DeiT avec Distillation"
            >
              <Cpu size={15} />
              <span>DeiT</span>
              {selectedModel === 'deit' && <span className="active-dot deit" />}
            </button>
          </div>

          <div className="control-pill-toggle">
            <span className="pill-toggle-label">Échelle des données :</span>
            <button
              type="button"
              className={`pill-toggle-btn scale-btn ${diffLevel === 'real' ? 'active' : ''}`}
              onClick={() => setDiffLevel('real')}
              title="Dimensions réelles ViT-Base (224×224, 196 patches, D=768)"
            >
              <span>Dimensions réelles</span>
            </button>
            <button
              type="button"
              className={`pill-toggle-btn scale-btn ${diffLevel === 'miniature' ? 'active' : ''}`}
              onClick={() => setDiffLevel('miniature')}
              title="Matrices réduites idéales pour calcul mental rapide"
            >
              <span>Miniature (pédagogique)</span>
            </button>
          </div>

          <div className={`active-model-status-chip ${selectedModel === 'deit' ? 'deit' : 'vit'}`}>
            <span className="chip-badge">{selectedModel === 'deit' ? 'Modèle DeiT sélectionné' : 'Modèle ViT sélectionné'}</span>
            <span className="chip-desc">
              {selectedModel === 'deit' ? 'Token [DIST] + Distillation CNN' : 'Classification supervisée [CLS]'}
            </span>
          </div>
        </div>
      </div>

      {/* 2. PROGRESSION & STATS BAR */}
      <div className="lab-progress-bar-card">
        <div className="lab-stats-left">
          <div className="lab-stat-item">
            <span className="stat-label">Progression du laboratoire :</span>
            <div className="stat-progress-pill">
              <div className="stat-track">
                <div className="stat-fill" style={{ width: `${progressPct}%` }} />
              </div>
              <span className="stat-pct-num">{completedCount} / 7 étapes ({progressPct}%)</span>
            </div>
          </div>

          <div className="lab-stat-item">
            <span className="stat-label">Score global :</span>
            <span className="score-badge">
              <Award size={14} />
              <b>{score} / 100 pts</b>
            </span>
          </div>

          <div className="lab-stat-item">
            <span className="stat-label">Tentatives :</span>
            <span className="attempts-badge">{attempts}</span>
          </div>
        </div>

        <button
          type="button"
          className="btn-reset-lab"
          onClick={() => {
            if (window.confirm('Voulez-vous réinitialiser vos scores et recommencer ?')) {
              resetAllProgress();
            }
          }}
          title="Réinitialiser le laboratoire"
        >
          <RotateCcw size={14} />
          <span>Réinitialiser</span>
        </button>
      </div>

      {/* 3. STEPPER EN 7 ÉTAPES */}
      <div className="math-stepper-container">
        <div className="math-stepper-track">
          {stepsList.map((st, idx) => {
            const isActive = currentStep === st.id;
            const isDone = validatedTheorySteps[st.id];
            return (
              <React.Fragment key={st.id}>
                <button
                  type="button"
                  className={`math-step-node ${isActive ? 'active' : ''} ${isDone ? 'past' : ''}`}
                  onClick={() => setCurrentStep(st.id)}
                >
                  <div className="step-num-bubble">
                    {isDone ? <Check size={14} strokeWidth={3} /> : st.id}
                  </div>
                  <span className="step-num-name">{st.title}</span>
                </button>
                {idx < stepsList.length - 1 && (
                  <div className={`math-step-line ${isDone ? 'filled' : ''}`} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* 4. CONTENU DE L'ÉTAPE ACTIVE */}
      <div className="lab-exercise-card">
        {/* ============================================================== */}
        {/* ÉTAPE 1 : DÉCOUPAGE EN PATCHES                                  */}
        {/* ============================================================== */}
        {currentStep === 1 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 01 / 07</span>
              <h3>Objectif : Calculer le découpage d'une image en patches et la taille aplatie</h3>
              <p>
                Une image 2D <InlineMath math="\mathbf{X} \in \mathbb{R}^{H \times W \times C}" /> est découpée en une
                grille régulière de <InlineMath math="N" /> patches carrés de côté <InlineMath math="P" />.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">1</span>
                  <span className="panel-header-title">Formules Mathématiques (KaTeX)</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="N = \frac{H \times W}{P^2} = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right),\quad D_{\text{flat}} = P^2 \times C" />
                </div>

                <div className="param-selector-row">
                  <span className="selector-label">Taille du patch <InlineMath math="P" /> :</span>
                  <div className="patch-size-btn-group">
                    {([8, 16, 32] as PatchSizeOption[]).map((p) => (
                      <button
                        key={p}
                        type="button"
                        className={`btn-patch-opt ${step1PatchSize === p ? 'active' : ''}`}
                        onClick={() => setStep1PatchSize(p)}
                      >
                        {p} × {p}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card">
                    <label>1. Nombre de patches par côté (<InlineMath math="H/P" />) :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step1Inputs.gridRowsCols}
                      onChange={(e) => setStep1Inputs({ ...step1Inputs, gridRowsCols: e.target.value })}
                      placeholder="Ex: 14"
                    />
                  </div>

                  <div className="input-field-card">
                    <label>2. Nombre total de patches (<InlineMath math="N" />) :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step1Inputs.totalPatches}
                      onChange={(e) => setStep1Inputs({ ...step1Inputs, totalPatches: e.target.value })}
                      placeholder="Ex: 196"
                    />
                  </div>

                  <div className="input-field-card">
                    <label>3. Dimension du vecteur aplati (<InlineMath math="D_{\text{flat}} = P^2 \times C" />) :</label>
                    <div className="input-with-unit">
                      <input
                        type="number"
                        className="styled-lab-input"
                        value={step1Inputs.flattenedDim}
                        onChange={(e) => setStep1Inputs({ ...step1Inputs, flattenedDim: e.target.value })}
                        placeholder="Ex: 768"
                      />
                      <span className="unit-label">valeurs</span>
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep1}>
                    <CheckCircle2 size={16} />
                    <span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-hint" onClick={() => showNextHint(1)}>
                    <HelpCircle size={15} />
                    <span>Indice {hintLevels[1] ? `(${hintLevels[1]}/3)` : ''}</span>
                  </button>
                  <button
                    type="button"
                    className="btn-toggle-solution"
                    onClick={() => setSolutionVisible({ ...solutionVisible, 1: !solutionVisible[1] })}
                  >
                    <span>{solutionVisible[1] ? 'Masquer la correction' : 'Voir la correction détaillée'}</span>
                  </button>
                </div>

                {step1Feedback && (
                  <div className={`feedback-alert-box ${step1Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step1Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step1Feedback.message}</span>
                  </div>
                )}

                {hintLevels[1] && hintLevels[1] >= 1 && (
                  <div className="hint-collapsible-box">
                    <strong>Indice {hintLevels[1]} :</strong>{' '}
                    {hintLevels[1] === 1 && `Chaque côté de 224 px est divisé en segments de ${step1PatchSize} px.`}
                    {hintLevels[1] === 2 && `Calculez (224 / ${step1PatchSize}) × (224 / ${step1PatchSize}) pour N.`}
                    {hintLevels[1] >= 3 && `Chaque patch contient ${step1PatchSize}×${step1PatchSize} pixels, chacun ayant 3 canaux (R, G, B). Multipliez par 3 !`}
                  </div>
                )}

                {solutionVisible[1] && (
                  <div className="solution-collapsible-box">
                    <h4>Correction détaillée pas à pas :</h4>
                    <p>
                      1. Nombre de patches par côté : <InlineMath math={`224 / ${step1PatchSize} = ${224 / step1PatchSize}`} />.
                    </p>
                    <p>
                      2. Nombre total de patches : <InlineMath math={`N = (${224 / step1PatchSize}) \\times (${224 / step1PatchSize}) = ${(224 / step1PatchSize) ** 2}`} />.
                    </p>
                    <p>
                      3. Vecteur aplati : <InlineMath math={`D = ${step1PatchSize}^2 \\times 3 = ${step1PatchSize * step1PatchSize * 3}`} /> valeurs.
                    </p>
                  </div>
                )}
              </div>

              {/* Dynamic Grid Visualizer */}
              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Visualisation dynamique de la grille</span>
                </div>

                <div className="dynamic-grid-viewport">
                  <div className="image-with-live-grid">
                    <img src={imageUrl} alt="Base" className="base-bird-image" />
                    <div
                      className="live-grid-svg-overlay"
                      style={{
                        display: 'grid',
                        gridTemplateColumns: `repeat(${224 / step1PatchSize}, 1fr)`,
                        gridTemplateRows: `repeat(${224 / step1PatchSize}, 1fr)`,
                      }}
                    >
                      {Array.from({ length: (224 / step1PatchSize) ** 2 }).map((_, i) => (
                        <div key={i} className="cell-overlay-rect" title={`Patch #${i + 1}`} />
                      ))}
                    </div>
                  </div>

                  <span className="grid-meta-summary">
                    Grille : <b>{224 / step1PatchSize} × {224 / step1PatchSize}</b> ({' '}
                    <b>{(224 / step1PatchSize) ** 2}</b> patches de {step1PatchSize}×{step1PatchSize} px )
                  </span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* ÉTAPE 2 : PROJECTION LINÉAIRE                                  */}
        {/* ============================================================== */}
        {currentStep === 2 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 02 / 07</span>
              <h3>Objectif : Calculer la projection linéaire affine du patch embedding</h3>
              <p>
                Chaque patch aplati <InlineMath math="\mathbf{x}_p \in \mathbb{R}^{D_{\text{in}}}" /> est projeté dans l'espace
                latent <InlineMath math="\mathbf{z} \in \mathbb{R}^{D_{\text{model}}}" /> via la matrice apprenable <InlineMath math="\mathbf{E}" />.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">2</span>
                  <span className="panel-header-title">Dimensions &amp; Multiplication Matricielle</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="\mathbf{z} = \mathbf{x}_p \mathbf{E} + \mathbf{b},\quad \mathbf{E} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{model}}}" />
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card">
                    <label>1. Dimension d'entrée <InlineMath math="D_{\text{in}}" /> :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step2Inputs.dimIn}
                      onChange={(e) => setStep2Inputs({ ...step2Inputs, dimIn: e.target.value })}
                      placeholder={diffLevel === 'miniature' ? '12' : '768'}
                    />
                  </div>

                  <div className="input-field-card">
                    <label>2. Dimension de sortie <InlineMath math="D_{\text{model}}" /> :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step2Inputs.dimOut}
                      onChange={(e) => setStep2Inputs({ ...step2Inputs, dimOut: e.target.value })}
                      placeholder={diffLevel === 'miniature' ? '4' : '768'}
                    />
                  </div>

                  <div className="sub-calc-step">
                    <span className="grp-label">
                      3. Multiplication manuelle sur un petit exemple :
                    </span>
                    <p className="desc-text">
                      Soit <InlineMath math="\mathbf{x} = [2, 1, 3]" />,{' '}
                      <InlineMath math="\mathbf{W} = \begin{pmatrix} 1 & 0 \\ 0 & 2 \\ 1 & 1 \end{pmatrix}" /> et{' '}
                      <InlineMath math="\mathbf{b} = [1, -1]" />. Calculez <InlineMath math="\mathbf{z} = \mathbf{x}\mathbf{W} + \mathbf{b}" /> :
                    </p>
                    <div className="result-cells-row">
                      <div className="vec-box">
                        <span>z₁ = </span>
                        <input
                          type="number"
                          className="cell-input"
                          value={step2Inputs.z1}
                          onChange={(e) => setStep2Inputs({ ...step2Inputs, z1: e.target.value })}
                          placeholder="?"
                        />
                      </div>
                      <div className="vec-box">
                        <span>z₂ = </span>
                        <input
                          type="number"
                          className="cell-input"
                          value={step2Inputs.z2}
                          onChange={(e) => setStep2Inputs({ ...step2Inputs, z2: e.target.value })}
                          placeholder="?"
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep2}>
                    <CheckCircle2 size={16} />
                    <span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-hint" onClick={() => showNextHint(2)}>
                    <HelpCircle size={15} />
                    <span>Indice {hintLevels[2] ? `(${hintLevels[2]}/3)` : ''}</span>
                  </button>
                  <button
                    type="button"
                    className="btn-toggle-solution"
                    onClick={() => setSolutionVisible({ ...solutionVisible, 2: !solutionVisible[2] })}
                  >
                    <span>{solutionVisible[2] ? 'Masquer la correction' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step2Feedback && (
                  <div className={`feedback-alert-box ${step2Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step2Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step2Feedback.message}</span>
                  </div>
                )}

                {solutionVisible[2] && (
                  <div className="solution-collapsible-box">
                    <h4>Calcul détaillé :</h4>
                    <p><InlineMath math="z_1 = 2(1) + 1(0) + 3(1) + 1 = 2 + 0 + 3 + 1 = 6" /></p>
                    <p><InlineMath math="z_2 = 2(0) + 1(2) + 3(1) - 1 = 0 + 2 + 3 - 1 = 4" /></p>
                    <p>Résultat : <InlineMath math="\mathbf{z} = [6, 4]" />.</p>
                  </div>
                )}
              </div>

              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Visualisation interactive des embeddings</span>
                </div>
                <div style={{ padding: '12px 10px' }}>
                  <EmbeddingRibbon patchSize={step1PatchSize} />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* ÉTAPE 3 : TOKENS & POSITIONS                                   */}
        {/* ============================================================== */}
        {currentStep === 3 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 03 / 07</span>
              <h3>Objectif : Concaténation des tokens spéciaux et somme positionnelle</h3>
              <p>
                ViT ajoute le token <code>[CLS]</code>. DeiT ajoute en plus le token <code>[DIST]</code>. On additionne ensuite l'encodage positionnel.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">3</span>
                  <span className="panel-header-title">Longueur de séquence &amp; Encodage</span>
                </div>

                <div className="formula-callout-box">
                  {selectedModel === 'deit' ? (
                    <BlockMath math="\mathbf{Z}_{\text{DeiT}} = \left[\mathbf{x}_{\text{cls}};\; \mathbf{x}_{\text{dist}};\; \mathbf{x}_p^1\mathbf{E};\; \dots;\; \mathbf{x}_p^N\mathbf{E}\right] + \mathbf{E}_{\text{pos}} \quad (N + 2 \text{ tokens})" />
                  ) : (
                    <BlockMath math="\mathbf{Z}_{\text{ViT}} = \left[\mathbf{x}_{\text{cls}};\; \mathbf{x}_p^1\mathbf{E};\; \dots;\; \mathbf{x}_p^N\mathbf{E}\right] + \mathbf{E}_{\text{pos}} \quad (N + 1 \text{ tokens})" />
                  )}
                </div>

                <div className="exercise-inputs-grid">
                  <div className={`input-field-card ${selectedModel === 'vit' ? 'highlighted-active-model' : ''}`}>
                    <label>
                      1. Longueur totale de séquence pour ViT ({diffLevel === 'miniature' ? '9' : '196'} patches) :
                      {selectedModel === 'vit' && <span className="active-model-tag">Actif</span>}
                    </label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step3Inputs.seqLenViT}
                      onChange={(e) => setStep3Inputs({ ...step3Inputs, seqLenViT: e.target.value })}
                      placeholder={diffLevel === 'miniature' ? '10' : '197'}
                    />
                  </div>

                  <div className={`input-field-card ${selectedModel === 'deit' ? 'highlighted-active-model deit' : ''}`}>
                    <label>
                      2. Longueur totale de séquence pour DeiT ({diffLevel === 'miniature' ? '9' : '196'} patches) :
                      {selectedModel === 'deit' && <span className="active-model-tag deit">Actif</span>}
                    </label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step3Inputs.seqLenDeiT}
                      onChange={(e) => setStep3Inputs({ ...step3Inputs, seqLenDeiT: e.target.value })}
                      placeholder={diffLevel === 'miniature' ? '11' : '198'}
                    />
                  </div>

                  <div className="sub-calc-step">
                    <span className="grp-label">3. Addition vectorielle d'encodage positionnel :</span>
                    <p className="desc-text">
                      Soit <InlineMath math="\mathbf{E} = [0.5, -0.2, 1.0]" /> et{' '}
                      <InlineMath math="\mathbf{E}_{\text{pos}} = [0.1, 0.4, -0.3]" />. Calculez <InlineMath math="\mathbf{Z} = \mathbf{E} + \mathbf{E}_{\text{pos}}" /> :
                    </p>
                    <div className="result-cells-row">
                      <div className="vec-box">
                        <span>Z₁ = </span>
                        <input
                          type="number"
                          step="0.1"
                          className="cell-input"
                          value={step3Inputs.zPos1}
                          onChange={(e) => setStep3Inputs({ ...step3Inputs, zPos1: e.target.value })}
                        />
                      </div>
                      <div className="vec-box">
                        <span>Z₂ = </span>
                        <input
                          type="number"
                          step="0.1"
                          className="cell-input"
                          value={step3Inputs.zPos2}
                          onChange={(e) => setStep3Inputs({ ...step3Inputs, zPos2: e.target.value })}
                        />
                      </div>
                      <div className="vec-box">
                        <span>Z₃ = </span>
                        <input
                          type="number"
                          step="0.1"
                          className="cell-input"
                          value={step3Inputs.zPos3}
                          onChange={(e) => setStep3Inputs({ ...step3Inputs, zPos3: e.target.value })}
                        />
                      </div>
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep3}>
                    <CheckCircle2 size={16} />
                    <span>Vérifier ma réponse</span>
                  </button>
                  <button
                    type="button"
                    className="btn-toggle-solution"
                    onClick={() => setSolutionVisible({ ...solutionVisible, 3: !solutionVisible[3] })}
                  >
                    <span>{solutionVisible[3] ? 'Masquer' : 'Voir la solution'}</span>
                  </button>
                </div>

                {step3Feedback && (
                  <div className={`feedback-alert-box ${step3Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step3Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step3Feedback.message}</span>
                  </div>
                )}
              </div>

              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Alignement des Tokens (Interactif)</span>
                </div>
                <div style={{ padding: '12px 10px' }}>
                  <TokenSequenceVisualizer
                    model={selectedModel as 'vit' | 'deit'}
                    patchSize={step1PatchSize}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* ÉTAPE 4 : LAYER NORMALIZATION                                   */}
        {/* ============================================================== */}
        {currentStep === 4 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 04 / 07</span>
              <h3>Objectif : Calculer la moyenne, la variance et la normalisation d'un vecteur</h3>
              <p>
                LayerNorm normalise chaque token selon <InlineMath math="\hat{x}_i = (x_i - \mu)/\sqrt{\sigma^2 + \epsilon}" />.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">4</span>
                  <span className="panel-header-title">Vecteur d'exemple : [1.0, 3.0, 3.0, 9.0]</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="\mu = \frac{1}{D}\sum_{i=1}^D x_i,\quad \sigma^2 = \frac{1}{D}\sum_{i=1}^D (x_i - \mu)^2,\quad \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}" />
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card">
                    <label>1. Moyenne <InlineMath math="\mu" /> :</label>
                    <input
                      type="number"
                      step="0.1"
                      className="styled-lab-input"
                      value={step4Inputs.mean}
                      onChange={(e) => setStep4Inputs({ ...step4Inputs, mean: e.target.value })}
                      placeholder="Ex: 4.0"
                    />
                  </div>

                  <div className="input-field-card">
                    <label>2. Variance <InlineMath math="\sigma^2" /> :</label>
                    <input
                      type="number"
                      step="0.1"
                      className="styled-lab-input"
                      value={step4Inputs.variance}
                      onChange={(e) => setStep4Inputs({ ...step4Inputs, variance: e.target.value })}
                      placeholder="Ex: 9.0"
                    />
                  </div>

                  <div className="input-field-card">
                    <label>3. Écart-type <InlineMath math="\sigma = \sqrt{\sigma^2}" /> :</label>
                    <input
                      type="number"
                      step="0.1"
                      className="styled-lab-input"
                      value={step4Inputs.stdDev}
                      onChange={(e) => setStep4Inputs({ ...step4Inputs, stdDev: e.target.value })}
                      placeholder="Ex: 3.0"
                    />
                  </div>

                  <div className="sub-calc-step">
                    <span className="grp-label">4. Vecteur normalisé <InlineMath math="\hat{\mathbf{x}}" /> :</span>
                    <div className="result-cells-row">
                      <input
                        type="number"
                        step="0.01"
                        className="cell-input"
                        value={step4Inputs.xNorm1}
                        onChange={(e) => setStep4Inputs({ ...step4Inputs, xNorm1: e.target.value })}
                        placeholder="x̂₁"
                      />
                      <input
                        type="number"
                        step="0.01"
                        className="cell-input"
                        value={step4Inputs.xNorm2}
                        onChange={(e) => setStep4Inputs({ ...step4Inputs, xNorm2: e.target.value })}
                        placeholder="x̂₂"
                      />
                      <input
                        type="number"
                        step="0.01"
                        className="cell-input"
                        value={step4Inputs.xNorm3}
                        onChange={(e) => setStep4Inputs({ ...step4Inputs, xNorm3: e.target.value })}
                        placeholder="x̂₃"
                      />
                      <input
                        type="number"
                        step="0.01"
                        className="cell-input"
                        value={step4Inputs.xNorm4}
                        onChange={(e) => setStep4Inputs({ ...step4Inputs, xNorm4: e.target.value })}
                        placeholder="x̂₄"
                      />
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep4}>
                    <CheckCircle2 size={16} />
                    <span>Vérifier ma réponse</span>
                  </button>
                  <button
                    type="button"
                    className="btn-toggle-solution"
                    onClick={() => setSolutionVisible({ ...solutionVisible, 4: !solutionVisible[4] })}
                  >
                    <span>{solutionVisible[4] ? 'Masquer' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step4Feedback && (
                  <div className={`feedback-alert-box ${step4Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step4Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step4Feedback.message}</span>
                  </div>
                )}
              </div>

              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Centrage &amp; Réduction visuelle</span>
                </div>
                <div className="norm-comparison-bars">
                  <div className="bar-visual-stack">
                    <div className="blue-fill" style={{ height: '70px' }} />
                    <span className="bar-title">Brut (μ=4, σ=3)</span>
                  </div>
                  <div className="bar-visual-stack">
                    <div className="green-fill" style={{ height: '70px' }} />
                    <span className="bar-title">Normalisé (μ=0, σ=1)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* ÉTAPE 5 : ATTENTION MULTI-HEAD                                  */}
        {/* ============================================================== */}
        {currentStep === 5 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 05 / 07</span>
              <h3>Objectif : Calculer le produit scalaire mis à l'échelle et le Softmax</h3>
              <p>
                On donne <InlineMath math="\mathbf{Q} = \begin{pmatrix} 2 & 0 \\ 1 & 1 \end{pmatrix}" /> et{' '}
                <InlineMath math="\mathbf{K} = \begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix}" /> avec <InlineMath math="d_k = 4" />.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">5</span>
                  <span className="panel-header-title">Équation de Scaled Dot-Product</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}},\quad \mathbf{A} = \text{softmax}(\mathbf{S})" />
                </div>

                <div className="exercise-inputs-grid">
                  <div className="sub-calc-step">
                    <span className="grp-label">1. Matrice transposée <InlineMath math="\mathbf{K}^T" /> :</span>
                    <div className="result-cells-row">
                      <input
                        type="number"
                        className="cell-input"
                        value={step5Inputs.kt11}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, kt11: e.target.value })}
                        placeholder="k₁₁"
                      />
                      <input
                        type="number"
                        className="cell-input"
                        value={step5Inputs.kt12}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, kt12: e.target.value })}
                        placeholder="k₁₂"
                      />
                    </div>
                    <div className="result-cells-row">
                      <input
                        type="number"
                        className="cell-input"
                        value={step5Inputs.kt21}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, kt21: e.target.value })}
                        placeholder="k₂₁"
                      />
                      <input
                        type="number"
                        className="cell-input"
                        value={step5Inputs.kt22}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, kt22: e.target.value })}
                        placeholder="k₂₂"
                      />
                    </div>
                  </div>

                  <div className="sub-calc-step">
                    <span className="grp-label">2. Scores échelonnés ligne 1 (<InlineMath math="\mathbf{Q}\mathbf{K}^T / \sqrt{4}" />) :</span>
                    <div className="result-cells-row">
                      <span>S₁₁ = </span>
                      <input
                        type="number"
                        step="0.1"
                        className="cell-input"
                        value={step5Inputs.s11}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, s11: e.target.value })}
                      />
                      <span>S₁₂ = </span>
                      <input
                        type="number"
                        step="0.1"
                        className="cell-input"
                        value={step5Inputs.s12}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, s12: e.target.value })}
                      />
                    </div>
                  </div>

                  <div className="sub-calc-step">
                    <span className="grp-label">3. Poids d'attention Softmax ligne 1 :</span>
                    <div className="result-cells-row">
                      <span>A₁₁ = </span>
                      <input
                        type="number"
                        step="0.01"
                        className="cell-input"
                        value={step5Inputs.a11}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, a11: e.target.value })}
                        placeholder="0.73"
                      />
                      <span>A₁₂ = </span>
                      <input
                        type="number"
                        step="0.01"
                        className="cell-input"
                        value={step5Inputs.a12}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, a12: e.target.value })}
                        placeholder="0.27"
                      />
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep5}>
                    <CheckCircle2 size={16} />
                    <span>Vérifier ma réponse</span>
                  </button>
                  <button
                    type="button"
                    className="btn-toggle-solution"
                    onClick={() => setSolutionVisible({ ...solutionVisible, 5: !solutionVisible[5] })}
                  >
                    <span>{solutionVisible[5] ? 'Masquer' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step5Feedback && (
                  <div className={`feedback-alert-box ${step5Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step5Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step5Feedback.message}</span>
                  </div>
                )}
              </div>

              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Heatmap d'attention 2×2</span>
                </div>
                <div className="mini-heatmap-grid">
                  <div className="heat-cell" style={{ backgroundColor: 'rgba(37, 99, 235, 0.73)' }}>
                    73%
                  </div>
                  <div className="heat-cell" style={{ backgroundColor: 'rgba(37, 99, 235, 0.27)' }}>
                    27%
                  </div>
                  <div className="heat-cell" style={{ backgroundColor: 'rgba(37, 99, 235, 0.50)' }}>
                    50%
                  </div>
                  <div className="heat-cell" style={{ backgroundColor: 'rgba(37, 99, 235, 0.50)' }}>
                    50%
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* ÉTAPE 6 : MLP & GELU                                           */}
        {/* ============================================================== */}
        {currentStep === 6 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 06 / 07</span>
              <h3>Objectif : Calculer l'expansion MLP et l'activation non linéaire GELU</h3>
              <p>
                L'activation <InlineMath math="\text{GELU}(x) = x \Phi(x)" /> pondère l'entrée par la probabilité cumulée d'une loi normale standard.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">6</span>
                  <span className="panel-header-title">Formule Exacte &amp; Approximation</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="\text{GELU}(x) \approx 0.5 x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}} (x + 0.044715 x^3)\right)\right)" />
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card">
                    <label>1. Facteur d'expansion de la couche cachée MLP :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step6Inputs.expansionFactor}
                      onChange={(e) => setStep6Inputs({ ...step6Inputs, expansionFactor: e.target.value })}
                      placeholder="Ex: 4"
                    />
                  </div>

                  <div className="input-field-card">
                    <label>2. Dimension MLP pour ViT-Base (<InlineMath math="4 \times 768" />) :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step6Inputs.mlpDim}
                      onChange={(e) => setStep6Inputs({ ...step6Inputs, mlpDim: e.target.value })}
                      placeholder="Ex: 3072"
                    />
                  </div>

                  <div className="sub-calc-step">
                    <span className="grp-label">3. Calcul de GELU :</span>
                    <div className="result-cells-row">
                      <span>GELU(3.0) ≈ </span>
                      <input
                        type="number"
                        step="0.001"
                        className="cell-input"
                        value={step6Inputs.geluZ1}
                        onChange={(e) => setStep6Inputs({ ...step6Inputs, geluZ1: e.target.value })}
                        placeholder="2.996"
                      />
                      <span>GELU(-1.0) ≈ </span>
                      <input
                        type="number"
                        step="0.001"
                        className="cell-input"
                        value={step6Inputs.geluZ2}
                        onChange={(e) => setStep6Inputs({ ...step6Inputs, geluZ2: e.target.value })}
                        placeholder="-0.159"
                      />
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep6}>
                    <CheckCircle2 size={16} />
                    <span>Vérifier ma réponse</span>
                  </button>
                  <button
                    type="button"
                    className="btn-toggle-solution"
                    onClick={() => setSolutionVisible({ ...solutionVisible, 6: !solutionVisible[6] })}
                  >
                    <span>{solutionVisible[6] ? 'Masquer' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step6Feedback && (
                  <div className={`feedback-alert-box ${step6Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step6Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step6Feedback.message}</span>
                  </div>
                )}
              </div>

              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Courbe GELU vs ReLU</span>
                </div>
                <div className="gelu-graph-container">
                  <svg className="gelu-svg-chart" viewBox="-3 -1 6 4">
                    <line x1="-3" y1="0" x2="3" y2="0" stroke="#cbd5e1" strokeWidth="0.05" />
                    <line x1="0" y1="-1" x2="0" y2="3" stroke="#cbd5e1" strokeWidth="0.05" />
                    {/* ReLU: y = max(0, x) */}
                    <path d="M -3 0 L 0 0 L 3 3" stroke="#94a3b8" strokeWidth="0.08" strokeDasharray="0.2 0.2" fill="none" />
                    {/* GELU smooth curve */}
                    <path
                      d="M -3 -0.01 Q -1 -0.2 0 0 Q 1 0.8 3 3"
                      stroke="#2563eb"
                      strokeWidth="0.12"
                      fill="none"
                    />
                  </svg>
                  <div className="graph-legend-row">
                    <span className="legend-gelu">— GELU (continue, non monotone)</span>
                    <span className="legend-relu">--- ReLU (brisée en 0)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ============================================================== */}
        {/* ÉTAPE 7 : CLASSIFICATION & DISTILLATION                        */}
        {/* ============================================================== */}
        {currentStep === 7 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 07 / 07</span>
              <h3>Objectif : Classification Softmax finale et fusion de distillation DeiT</h3>
              <p>
                Logits donnés : <InlineMath math="\mathbf{z} = [2.0, 1.0, 0.0]" /> pour 3 classes candidates.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">7</span>
                  <span className="panel-header-title">Formulation Softmax &amp; Fusion DeiT</span>
                </div>

                <div className="formula-callout-box">
                  {selectedModel === 'deit' ? (
                    <BlockMath math="P(y=i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}},\quad P_{\text{DeiT}}^{\text{final}} = \frac{1}{2}\left(P_{\text{cls}} + P_{\text{dist}}\right)" />
                  ) : (
                    <BlockMath math="P_{\text{ViT}}(y=i) = \frac{e^{z_i^{\text{cls}}}}{\sum_{j} e^{z_j^{\text{cls}}}},\quad \hat{y} = \operatorname{argmax}_i P_{\text{ViT}}(y=i)" />
                  )}
                </div>

                <div className="exercise-inputs-grid">
                  <div className="sub-calc-step">
                    <span className="grp-label">1. Exponentielles des logits :</span>
                    <div className="result-cells-row">
                      <span>exp(2.0) = </span>
                      <input
                        type="number"
                        step="0.001"
                        className="cell-input"
                        value={step7Inputs.exp1}
                        onChange={(e) => setStep7Inputs({ ...step7Inputs, exp1: e.target.value })}
                        placeholder="7.389"
                      />
                      <span>Somme = </span>
                      <input
                        type="number"
                        step="0.01"
                        className="cell-input"
                        value={step7Inputs.sumExp}
                        onChange={(e) => setStep7Inputs({ ...step7Inputs, sumExp: e.target.value })}
                        placeholder="11.11"
                      />
                    </div>
                  </div>

                  <div className="input-field-card">
                    <label>2. Probabilité de la classe 1 (<InlineMath math="P(y=1)" />) :</label>
                    <input
                      type="number"
                      step="0.01"
                      className="styled-lab-input"
                      value={step7Inputs.prob1}
                      onChange={(e) => setStep7Inputs({ ...step7Inputs, prob1: e.target.value })}
                      placeholder="0.665 (66.5%)"
                    />
                  </div>

                  {selectedModel === 'deit' ? (
                    <div className="sub-calc-step deit-highlight-box">
                      <span className="grp-label">
                        3. Fusion des têtes DeiT (<InlineMath math="P_{\text{cls}} = [0.60, 0.30, 0.10]" />, <InlineMath math="P_{\text{dist}} = [0.80, 0.15, 0.05]" />) :
                      </span>
                      <p className="desc-text">
                        Calculez la moyenne arithmétique de distillation : <InlineMath math="P_i = \frac{1}{2}(P_{\text{cls},i} + P_{\text{dist},i})" />
                      </p>
                      <div className="result-cells-row">
                        <span>P₁ = </span>
                        <input
                          type="number"
                          step="0.01"
                          className="cell-input"
                          value={step7Inputs.pFinal1}
                          onChange={(e) => setStep7Inputs({ ...step7Inputs, pFinal1: e.target.value })}
                          placeholder="0.70"
                        />
                        <span>P₂ = </span>
                        <input
                          type="number"
                          step="0.001"
                          className="cell-input"
                          value={step7Inputs.pFinal2}
                          onChange={(e) => setStep7Inputs({ ...step7Inputs, pFinal2: e.target.value })}
                          placeholder="0.225"
                        />
                      </div>
                    </div>
                  ) : (
                    <div className="vit-head-notice-box">
                      <span className="notice-pill">Architecture ViT (Tête Unique [CLS])</span>
                      <p className="desc-text">
                        Dans Google ViT, la décision finale repose à <strong>100% sur le token [CLS]</strong> : <InlineMath math="\hat{y} = \operatorname{argmax}_i P(y=i) = 1" /> (classe 1 gagne avec ~66.5% de confiance).
                        Il n'y a pas de token [DIST] ni de seconde tête de distillation. Pour vous entraîner sur la fusion de distillation, sélectionnez <strong>DeiT</strong> dans le sélecteur d'architecture.
                      </p>
                    </div>
                  )}
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep7}>
                    <CheckCircle2 size={16} />
                    <span>Vérifier ma réponse</span>
                  </button>
                  <button
                    type="button"
                    className="btn-toggle-solution"
                    onClick={() => setSolutionVisible({ ...solutionVisible, 7: !solutionVisible[7] })}
                  >
                    <span>{solutionVisible[7] ? 'Masquer' : 'Voir la solution'}</span>
                  </button>
                </div>

                {step7Feedback && (
                  <div className={`feedback-alert-box ${step7Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step7Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step7Feedback.message}</span>
                  </div>
                )}
              </div>

              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Récapitulatif End-to-End</span>
                </div>
                <div className="pipeline-recap-vertical">
                  <div className="recap-node">Image brute (224×224×3)</div>
                  <div className="recap-arr">↓</div>
                  <div className="recap-node">196 patches (16×16)</div>
                  <div className="recap-arr">↓</div>
                  <div className="recap-node">Projection linéaire (768D)</div>
                  <div className="recap-arr">↓</div>
                  <div className="recap-node">Tokens [CLS] + [DIST] + Position</div>
                  <div className="recap-arr">↓</div>
                  <div className="recap-node">12 Blocs Pre-LN Transformer</div>
                  <div className="recap-arr">↓</div>
                  <div className="recap-node highlight">Classification finale ImageNet (1000 classes)</div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
