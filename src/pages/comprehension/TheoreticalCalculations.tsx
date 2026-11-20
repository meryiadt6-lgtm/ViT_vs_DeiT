import React, { useState, useMemo } from 'react';
import {
  Calculator,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  XCircle,
  HelpCircle,
  RotateCcw,
  Eye,
  Layers,
  Award,
  Check,
  Cpu,
  Brain,
  Info,
  GitBranch,
  Zap,
  BookOpen,
  AlertCircle,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import { InlineMath, BlockMath } from '../../components/MathFormula';
import {
  calculatePatchStats,
  getModeConfig,
  getSeqLen,
  isClose,
  calculateLayerNorm,
  geluExact,
  calculateSoftmax,
  calculateAttentionFixed,
  getStep7References,
} from './lab/mathEngine';
import { PatchSizeOption, ValidationFeedback } from './lab/types';

// ─────────────────────────────────────────────────────────────────────────────
// EmbeddingRibbon
// ─────────────────────────────────────────────────────────────────────────────
function EmbeddingRibbon({ N, dModel }: { N: number; dModel: number }) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  const getBarColor = (i: number) => {
    const hue = (i * 37 + 200) % 360;
    if (selectedIdx === i) return `hsl(${hue}, 90%, 45%)`;
    if (hoveredIdx === i) return `hsl(${hue}, 80%, 55%)`;
    return `hsl(${hue}, 60%, 65%)`;
  };

  const visibleBars = Math.min(16, N);
  const showDots = N > 17;

  return (
    <div className="embed-ribbon-container">
      <div className="embed-ribbon-header">
        <span className="embed-ribbon-title">
          Ruban de <strong>{N}</strong> vecteurs latents
          <span className="embed-ribbon-dim-badge">dim = {dModel}</span>
        </span>
        <span className="embed-ribbon-hint">Survolez ou cliquez un patch</span>
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
            title={`Patch #${i + 1} → vecteur ℝ^${dModel}`}
          />
        ))}
        {showDots && <span className="embed-vbar-dots">···</span>}
        {N > visibleBars && (
          <div
            key="last"
            className={`embed-vbar ${selectedIdx === N - 1 ? 'selected' : ''}`}
            style={{ background: getBarColor(N - 1) }}
            onMouseEnter={() => setHoveredIdx(N - 1)}
            onMouseLeave={() => setHoveredIdx(null)}
            onClick={() => setSelectedIdx(selectedIdx === N - 1 ? null : N - 1)}
            title={`Patch #${N} → vecteur ℝ^${dModel}`}
          />
        )}
      </div>

      {(hoveredIdx !== null || selectedIdx !== null) && (() => {
        const idx = hoveredIdx ?? selectedIdx!;
        const gridW = Math.round(Math.sqrt(N));
        const row = Math.floor(idx / gridW) + 1;
        const col = (idx % gridW) + 1;
        return (
          <div className="embed-detail-card">
            <div className="embed-detail-header">
              <span className="embed-detail-pill">Patch #{idx + 1}</span>
              <span className="embed-detail-pos">Ligne {row}, Col {col}</span>
            </div>
            <div className="embed-detail-body">
              <div className="embed-detail-row">
                <span className="embed-detail-label">Entrée (aplatie)</span>
                <span className="embed-detail-val">
                  <InlineMath math={`\\mathbf{x}_p \\in \\mathbb{R}^{${N <= 10 ? Math.round(Math.sqrt(N)) * Math.round(Math.sqrt(N)) * 3 : 768}}`} />
                </span>
              </div>
              <div className="embed-detail-row">
                <span className="embed-detail-label">Embedding (sortie)</span>
                <span className="embed-detail-val">
                  <InlineMath math={`\\mathbf{z}_{p,${idx+1}} \\in \\mathbb{R}^{${dModel}}`} />
                </span>
              </div>
              <div className="embed-minibar-vis">
                {Array.from({ length: 12 }).map((_, j) => {
                  const v = Math.abs(Math.sin((idx + 1) * 7.3 + j * 2.1));
                  return (
                    <div
                      key={j}
                      className="embed-minibar-cell"
                      style={{ height: `${12 + v * 24}px`, opacity: 0.5 + v * 0.5 }}
                      title={`dim ${j + 1}: ~${v.toFixed(3)}`}
                    />
                  );
                })}
                <span className="embed-minibar-dots">···{dModel}</span>
              </div>
            </div>
          </div>
        );
      })()}

      <div className="embed-ribbon-stats">
        <div className="embed-stat-box"><span className="embed-stat-num">{N}</span><span className="embed-stat-lbl">patches</span></div>
        <div className="embed-stat-box"><span className="embed-stat-num">×</span><span className="embed-stat-lbl" /></div>
        <div className="embed-stat-box blue"><span className="embed-stat-num">{dModel}</span><span className="embed-stat-lbl">dim. sortie</span></div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// TokenSequenceVisualizer
// ─────────────────────────────────────────────────────────────────────────────
function TokenSequenceVisualizer({ model, N, dModel }: { model: 'vit' | 'deit'; N: number; dModel: number }) {
  const [hoveredTok, setHoveredTok] = useState<string | null>(null);
  const seqLen = getSeqLen(N, model);

  const tokenDefs = [
    {
      id: 'cls', label: '[CLS]', color: '#1e40af', bg: '#eff6ff', border: '#93c5fd',
      desc: 'Token de classification. Son embedding appris est concaténé en tête de séquence. Après les L blocs encodeurs, son vecteur final est donné à la tête de classification.',
    },
    ...(model === 'deit' ? [{
      id: 'dist', label: '[DIST]', color: '#6d28d9', bg: '#f5f3ff', border: '#c4b5fd',
      desc: 'Token de distillation (DeiT uniquement). Embedding appris séparé. Sa sortie finale est donnée à la tête de distillation qui imite le modèle enseignant CNN (RegNetY-16GF).',
    }] : []),
    {
      id: 'p1', label: 'z_p¹', color: '#065f46', bg: '#ecfdf5', border: '#6ee7b7',
      desc: `Premier patch embedding z_p¹ ∈ ℝ^${dModel}. Représentation vectorielle du patch en haut à gauche de l'image après projection linéaire.`,
    },
    {
      id: 'p2', label: 'z_p²', color: '#065f46', bg: '#ecfdf5', border: '#6ee7b7',
      desc: 'Deuxième patch embedding (patch suivant dans la séquence de gauche à droite, haut en bas).',
    },
    {
      id: 'dots', label: '···', color: '#64748b', bg: '#f8fafc', border: '#e2e8f0',
      desc: `${N - 2} autres patch embeddings (z_p³ à z_p^${N-1})`,
    },
    {
      id: `pN`, label: `z_p^${N}`, color: '#065f46', bg: '#ecfdf5', border: '#6ee7b7',
      desc: `Dernier patch embedding (patch en bas à droite). Total : ${N} patches + ${model === 'deit' ? 2 : 1} token(s) spéciaux = ${seqLen} tokens.`,
    },
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
          <span className="tokseq-pos-label">Embedding du patch :</span>
          <span className="tokseq-vec-example">
            <InlineMath math="\mathbf{z}_p = [0.5,\; -0.2,\; 1.0]" />
          </span>
        </div>
        <div className="tokseq-pos-row">
          <span className="tokseq-pos-label">Encodage positionnel :</span>
          <span className="tokseq-vec-example">
            <InlineMath math="\mathbf{e}_{\text{pos}} = [0.1,\; 0.4,\; -0.3]" />
          </span>
        </div>
        <div className="tokseq-pos-row result">
          <span className="tokseq-pos-label">Résultat <InlineMath math="\mathbf{Z}" /> :</span>
          <span className="tokseq-vec-example green">
            <InlineMath math="\mathbf{Z} = [0.6,\; 0.2,\; 0.7]" />
          </span>
        </div>
        <p style={{ fontSize: '11px', color: '#64748b', marginTop: '6px', lineHeight: 1.5 }}>
          <strong>Note :</strong> <InlineMath math="\mathbf{E}_{\text{pos}}" /> contient {seqLen} lignes (une par token, y compris [CLS]{model === 'deit' ? ' et [DIST]' : ''}). Chaque token a sa propre position apprise.
        </p>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// GELU SVG Chart (corrected: y-axis not inverted, slight dip near x≈−0.75)
// ─────────────────────────────────────────────────────────────────────────────
function GELUChart() {
  // Generate GELU points from x=-3 to x=3
  const pts: [number, number][] = [];
  for (let xi = -3; xi <= 3; xi += 0.1) {
    const c = Math.sqrt(2 / Math.PI);
    const y = 0.5 * xi * (1 + Math.tanh(c * (xi + 0.044715 * xi * xi * xi)));
    pts.push([xi, y]);
  }

  // SVG: viewBox x from -3 to 3, y from -0.5 to 3.2
  // Map: svgX = (x+3) * 50, svgY = (3.2-y) * 50
  const mapX = (x: number) => (x + 3) * 50;
  const mapY = (y: number) => (3.2 - y) * 50;

  const geluPath = pts.map(([x, y], i) =>
    `${i === 0 ? 'M' : 'L'} ${mapX(x).toFixed(1)} ${mapY(y).toFixed(1)}`
  ).join(' ');

  // ReLU path: y=max(0,x)
  const reluPath = `M ${mapX(-3)} ${mapY(0)} L ${mapX(0)} ${mapY(0)} L ${mapX(3)} ${mapY(3)}`;

  const W = 300, H = 185;

  // Axis origin
  const ox = mapX(0), oy = mapY(0);

  return (
    <div className="gelu-graph-container">
      <svg width={W} height={H} viewBox={`0 0 ${W} ${H}`} style={{ overflow: 'visible' }}>
        {/* Grid lines */}
        {[-2, -1, 1, 2, 3].map(v => (
          <line key={`vg${v}`} x1={mapX(v)} y1={0} x2={mapX(v)} y2={H} stroke="#f1f5f9" strokeWidth="1" />
        ))}
        {[-0.5, 0.5, 1, 1.5, 2, 2.5, 3].map(v => (
          <line key={`hg${v}`} x1={0} y1={mapY(v)} x2={W} y2={mapY(v)} stroke="#f1f5f9" strokeWidth="1" />
        ))}
        {/* Axes */}
        <line x1={0} y1={oy} x2={W} y2={oy} stroke="#94a3b8" strokeWidth="1.5" />
        <line x1={ox} y1={0} x2={ox} y2={H} stroke="#94a3b8" strokeWidth="1.5" />
        {/* Axis labels */}
        {[-2, -1, 1, 2, 3].map(v => (
          <text key={`xl${v}`} x={mapX(v)} y={oy + 14} textAnchor="middle" fontSize="9" fill="#94a3b8">{v}</text>
        ))}
        {[1, 2, 3].map(v => (
          <text key={`yl${v}`} x={ox - 8} y={mapY(v) + 3} textAnchor="end" fontSize="9" fill="#94a3b8">{v}</text>
        ))}
        <text x={ox - 8} y={mapY(-0.17) + 3} textAnchor="end" fontSize="8" fill="#f59e0b">−0.17</text>
        {/* ReLU dashed */}
        <path d={reluPath} stroke="#94a3b8" strokeWidth="1.8" strokeDasharray="5 4" fill="none" opacity="0.7" />
        {/* GELU solid */}
        <path d={geluPath} stroke="#2563eb" strokeWidth="2.2" fill="none" />
        {/* Annotation points */}
        <circle cx={mapX(3)} cy={mapY(geluExact(3))} r="4" fill="#2563eb" />
        <text x={mapX(3) - 5} y={mapY(geluExact(3)) - 7} fontSize="9" fill="#1d4ed8" textAnchor="middle">2.996</text>
        <circle cx={mapX(-1)} cy={mapY(geluExact(-1))} r="4" fill="#ef4444" />
        <text x={mapX(-1) + 24} y={mapY(geluExact(-1)) + 4} fontSize="9" fill="#dc2626">−0.159</text>
        {/* Dip annotation near x=-0.75 */}
        <circle cx={mapX(-0.75)} cy={mapY(geluExact(-0.75))} r="3" fill="#f59e0b" />
        {/* Axis labels */}
        <text x={W - 8} y={oy - 5} fontSize="10" fill="#64748b" fontWeight="600">x</text>
        <text x={ox + 5} y={8} fontSize="10" fill="#64748b" fontWeight="600">y</text>
      </svg>
      <div className="graph-legend-row">
        <span className="legend-gelu">— GELU (continue, creux ≈ −0.17 vers x≈−0.75)</span>
        <span className="legend-relu">--- ReLU (brisée en 0)</span>
      </div>
    </div>
  );
}

function geluExactLocal(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * x * x * x)));
}

// ─────────────────────────────────────────────────────────────────────────────
// EncoderBlockDiagram – visual synthesis after step 6
// ─────────────────────────────────────────────────────────────────────────────
function EncoderBlockDiagram() {
  return (
    <div className="encoder-block-diagram">
      <div className="enc-block-title">Structure d'un bloc encodeur pré-LN (×12)</div>
      <div className="enc-block-flow">
        {/* Input */}
        <div className="enc-node input-node">
          <span>z<sub>ℓ-1</sub></span>
        </div>
        <div className="enc-arrow">↓</div>

        {/* Residual branch 1 */}
        <div className="enc-residual-group">
          <div className="enc-residual-label">Résidu 1 : MHSA</div>
          <div className="enc-block-row">
            <div className="enc-node ln-node">LN<br/><span style={{fontSize:'9px'}}>Étape 4</span></div>
            <span className="enc-mini-arrow">→</span>
            <div className="enc-node attn-node">MSA<br/><span style={{fontSize:'9px'}}>Étape 5</span></div>
          </div>
          <div className="enc-residual-formula">
            <InlineMath math="\mathbf{z}' = \text{MSA}(\text{LN}(\mathbf{z}_{\ell-1})) + \mathbf{z}_{\ell-1}" />
          </div>
        </div>
        <div className="enc-arrow">↓</div>

        {/* Residual branch 2 */}
        <div className="enc-residual-group">
          <div className="enc-residual-label">Résidu 2 : MLP</div>
          <div className="enc-block-row">
            <div className="enc-node ln-node">LN<br/><span style={{fontSize:'9px'}}>Étape 4</span></div>
            <span className="enc-mini-arrow">→</span>
            <div className="enc-node mlp-node">MLP<br/><span style={{fontSize:'9px'}}>Étape 6</span></div>
          </div>
          <div className="enc-residual-formula">
            <InlineMath math="\mathbf{z}_{\ell} = \text{MLP}(\text{LN}(\mathbf{z}')) + \mathbf{z}'" />
          </div>
        </div>
        <div className="enc-arrow">↓</div>

        {/* Output */}
        <div className="enc-node output-node">
          <span>z<sub>ℓ</sub></span>
        </div>

        <div className="enc-repeat-badge">× L = 12</div>
      </div>
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// End-to-End Pipeline – model-aware dual branches
// ─────────────────────────────────────────────────────────────────────────────
function PipelineRecap({ model }: { model: 'vit' | 'deit' }) {
  const vitNodes = [
    'Image 224×224×3',
    '196 patches 16×16',
    'Projection linéaire (768D)',
    '[CLS] + 196 z_p + E_pos → 197 tokens',
    '12 blocs pré-LN (LN→MSA→res→LN→MLP→res)',
    'LN final sur token [CLS]',
    'Tête linéaire 768→1000',
    'Softmax → argmax → classe',
  ];
  const deitNodes = [
    'Image 224×224×3',
    '196 patches 16×16',
    'Projection linéaire (768D)',
    '[CLS] + [DIST] + 196 z_p + E_pos → 198 tokens',
    '12 blocs pré-LN (LN→MSA→res→LN→MLP→res)',
    'LN final sur [CLS] et [DIST]',
    'Tête CLS (768→1000) + Tête DIST (768→1000)',
    'Fusion : P_final = ½(P_cls + P_dist)',
    'argmax P_final → classe',
  ];
  const nodes = model === 'deit' ? deitNodes : vitNodes;
  const color = model === 'deit' ? '#6d28d9' : '#1e3a8a';

  return (
    <div className="pipeline-recap-vertical">
      <div className="pipeline-model-badge" style={{ background: model === 'deit' ? '#f5f3ff' : '#eff6ff', color }}>
        {model === 'vit' ? '🧠 ViT — Classification [CLS] uniquement' : '🤖 DeiT — Classification + Distillation [DIST]'}
      </div>
      {nodes.map((n, i) => (
        <React.Fragment key={i}>
          <div className={`recap-node ${i === nodes.length - 1 ? 'highlight' : ''}`} style={i === nodes.length - 1 ? { borderColor: color, color } : {}}>
            {n}
          </div>
          {i < nodes.length - 1 && <div className="recap-arr">↓</div>}
        </React.Fragment>
      ))}
    </div>
  );
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────
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

  // Derive config from diffLevel
  const cfg = useMemo(() => getModeConfig(diffLevel as 'miniature' | 'real'), [diffLevel]);
  const isReal = diffLevel === 'real';

  // ── Step 1 ──
  const [step1PatchSize, setStep1PatchSize] = useState<PatchSizeOption>(16);
  const [step1Inputs, setStep1Inputs] = useState({ gridRowsCols: '', totalPatches: '', flattenedDim: '' });
  const [step1Feedback, setStep1Feedback] = useState<ValidationFeedback | null>(null);

  // ── Step 2 ──
  const [step2Inputs, setStep2Inputs] = useState({ dimIn: '', dimOut: '', totalParams: '', z1: '', z2: '' });
  const [step2Feedback, setStep2Feedback] = useState<ValidationFeedback | null>(null);

  // ── Step 3 ──
  const [step3Inputs, setStep3Inputs] = useState({ seqLen: '', zPos1: '', zPos2: '', zPos3: '' });
  const [step3Feedback, setStep3Feedback] = useState<ValidationFeedback | null>(null);

  // ── Step 4 ──
  const [step4Inputs, setStep4Inputs] = useState({ mean: '', variance: '', stdDev: '', xNorm1: '', xNorm2: '', xNorm3: '', xNorm4: '' });
  const [step4Feedback, setStep4Feedback] = useState<ValidationFeedback | null>(null);

  // ── Step 5 ──
  const [step5Inputs, setStep5Inputs] = useState({
    kt11: '', kt12: '', kt21: '', kt22: '',
    qkt11: '', qkt12: '', qkt21: '', qkt22: '',
    s11: '', s12: '',
    a11: '', a12: '',
  });
  const [step5Feedback, setStep5Feedback] = useState<ValidationFeedback | null>(null);

  // ── Step 6 ──
  const [step6Inputs, setStep6Inputs] = useState({ expansionFactor: '', mlpDim: '', geluZ1: '', geluZ2: '' });
  const [step6Feedback, setStep6Feedback] = useState<ValidationFeedback | null>(null);

  // ── Step 7 ──
  const [step7Inputs, setStep7Inputs] = useState({
    exp1: '', sumExp: '', prob1: '',
    pFinalPaper1: '', pFinalPaper2: '', pFinalPaper3: '',
  });
  const [step7Feedback, setStep7Feedback] = useState<ValidationFeedback | null>(null);

  const stepsList = [
    { id: 1, title: 'Découpage' },
    { id: 2, title: 'Projection' },
    { id: 3, title: 'Tokens & pos.' },
    { id: 4, title: 'LayerNorm' },
    { id: 5, title: 'Attention Q,K,V' },
    { id: 6, title: 'MLP & GELU' },
    { id: 7, title: 'Classification' },
  ];

  const recordValidation = (stepId: number, isSuccess: boolean) => {
    registerAttempt();
    if (isSuccess) validateTheoryStep(stepId, 15);
  };

  const showNextHint = (stepId: number) => {
    setHintLevels(prev => ({ ...prev, [stepId]: Math.min(3, (prev[stepId] || 0) + 1) }));
  };

  // ── Reference computations ──
  const patchStats = useMemo(() => {
    if (isReal) return calculatePatchStats(224, 224, step1PatchSize, 3);
    return calculatePatchStats(6, 6, 2, 3);
  }, [isReal, step1PatchSize]);

  const seqLenExpected = useMemo(
    () => getSeqLen(isReal ? patchStats.totalPatches : 9, selectedModel as 'vit' | 'deit'),
    [isReal, patchStats.totalPatches, selectedModel]
  );

  const lnRef = useMemo(() => calculateLayerNorm([1, 3, 3, 9]), []);
  const attnRef = useMemo(() => calculateAttentionFixed(), []);
  const step7Ref = useMemo(() => getStep7References(), []);

  // ─────────────────────────────────────────────────────────────
  // VERIFICATIONS
  // ─────────────────────────────────────────────────────────────
  const verifyStep1 = () => {
    const stats = isReal
      ? calculatePatchStats(224, 224, step1PatchSize, 3)
      : calculatePatchStats(6, 6, 2, 3);
    const H = isReal ? 224 : 6;
    const P = isReal ? step1PatchSize : 2;

    const okGrid = isClose(step1Inputs.gridRowsCols, stats.gridRows, 0.1);
    const okTotal = isClose(step1Inputs.totalPatches, stats.totalPatches, 0.1);
    const okDim = isClose(step1Inputs.flattenedDim, stats.flattenedDim, 0.1);
    const isSuccess = okGrid && okTotal && okDim;
    recordValidation(1, isSuccess);

    setStep1Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? `Parfait ! Pour une image ${H}×${H} avec P=${P} : ${stats.gridRows} patches par côté, N=${stats.totalPatches} patches au total, D_flat=${stats.flattenedDim} valeurs.`
        : `Pour ${H}×${H} avec P=${P} : lignes/cols = ${H}/${P} = ${stats.gridRows}, N = ${stats.gridRows}² = ${stats.totalPatches}, D_flat = ${P}²×3 = ${stats.flattenedDim}.`,
    });
  };

  const verifyStep2 = () => {
    // Dimensions selon mode
    const expectedDin = isReal ? 768 : 12;
    const expectedDout = isReal ? 768 : 4;
    // Params = D_in × D_model + D_model (biais, implémentation HuggingFace)
    const expectedParams = isReal ? 768 * 768 + 768 : 12 * 4 + 4;

    const okDin = isClose(step2Inputs.dimIn, expectedDin, 0.1);
    const okDout = isClose(step2Inputs.dimOut, expectedDout, 0.1);
    // Mini calc: x=[2,1,3], W=[[1,0],[0,2],[1,1]], b=[1,-1] → z=[6,4]
    const okZ1 = isClose(step2Inputs.z1, 6, 0.1);
    const okZ2 = isClose(step2Inputs.z2, 4, 0.1);
    const isSuccess = okDin && okDout && okZ1 && okZ2;
    recordValidation(2, isSuccess);

    setStep2Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? `Bravo ! D_in=${expectedDin}, D_model=${expectedDout}. z₁ = 2(1)+1(0)+3(1)+1 = 6, z₂ = 2(0)+1(2)+3(1)−1 = 4.`
        : `D_in = ${expectedDin}, D_model = ${expectedDout}. Exemple : z₁ = 2×1 + 1×0 + 3×1 + 1 = 6, z₂ = 2×0 + 1×2 + 3×1 − 1 = 4.`,
    });
  };

  const verifyStep3 = () => {
    const expectedLen = seqLenExpected;
    const okLen = isClose(step3Inputs.seqLen, expectedLen, 0.1);
    // z_p=[0.5,−0.2,1.0] + e_pos=[0.1,0.4,−0.3] = [0.6, 0.2, 0.7]
    const okZ1 = isClose(step3Inputs.zPos1, 0.6, 0.02);
    const okZ2 = isClose(step3Inputs.zPos2, 0.2, 0.02);
    const okZ3 = isClose(step3Inputs.zPos3, 0.7, 0.02);
    const isSuccess = okLen && okZ1 && okZ2 && okZ3;
    recordValidation(3, isSuccess);

    const N = isReal ? patchStats.totalPatches : 9;
    setStep3Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? `Parfait ! ${selectedModel === 'vit' ? `ViT : N+1 = ${N}+1 = ${expectedLen} tokens` : `DeiT : N+2 = ${N}+2 = ${expectedLen} tokens`}. z_p + e_pos = [0.6, 0.2, 0.7].`
        : `${selectedModel === 'vit' ? `ViT : longueur = ${N}+1 = ${expectedLen}` : `DeiT : longueur = ${N}+2 = ${expectedLen}`}. z_p=[0.5,−0.2,1.0] + e_pos=[0.1,0.4,−0.3] = [0.6, 0.2, 0.7].`,
    });
  };

  const verifyStep4 = () => {
    // [1,3,3,9] : μ=4, σ²=9, σ=3, x̂=[−1, −0.333, −0.333, 1.667]
    const okMean = isClose(step4Inputs.mean, 4.0, 0.01);
    const okVar = isClose(step4Inputs.variance, 9.0, 0.01);
    const okStd = isClose(step4Inputs.stdDev, 3.0, 0.01);
    const okN1 = isClose(step4Inputs.xNorm1, -1.0, 0.01);
    const okN2 = isClose(step4Inputs.xNorm2, -0.333, 0.01);
    const okN3 = isClose(step4Inputs.xNorm3, -0.333, 0.01);
    const okN4 = isClose(step4Inputs.xNorm4, 1.667, 0.01);
    const isSuccess = okMean && okVar && okStd && okN1 && okN2 && okN3 && okN4;
    recordValidation(4, isSuccess);

    setStep4Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? 'Superbe ! μ=4, σ²=9, σ=3. x̂=[−1.000, −0.333, −0.333, 1.667]. Layer Normalization stabilise les gradients.'
        : 'μ=(1+3+3+9)/4=4, σ²=((1−4)²+(3−4)²+(3−4)²+(9−4)²)/4=36/4=9, σ=3. x̂₁=(1−4)/3=−1.000, x̂₄=(9−4)/3=1.667.',
    });
  };

  const verifyStep5 = () => {
    // Corrected: Q=[[2,0,1,0],[1,1,0,1]], K=[[1,1,0,0],[0,2,1,0]], dk=4
    // KT cols: [1,0],[1,2],[0,1],[0,0]  → 4×2 matrix
    // KT[0]=[1,0], KT[1]=[1,2], KT[2]=[0,1], KT[3]=[0,0]
    // But we show 2×2 of KT for user: the first 2 rows
    // QKT = [[2,1],[2,2]]
    // S = [[1, 0.5],[1, 1]]
    // A row0 = softmax([1, 0.5]) ≈ [0.622, 0.378]
    // A row1 = softmax([1, 1]) = [0.5, 0.5]
    const ref = attnRef;

    // We ask user for QKT (first row only for simplicity), S row1, A row1
    const okQkt11 = isClose(step5Inputs.qkt11, ref.QKT[0][0], 0.1);
    const okQkt12 = isClose(step5Inputs.qkt12, ref.QKT[0][1], 0.1);
    const okS11 = isClose(step5Inputs.s11, ref.S[0][0], 0.01);
    const okS12 = isClose(step5Inputs.s12, ref.S[0][1], 0.01);
    const okA11 = isClose(step5Inputs.a11, ref.A[0][0], 0.01);
    const okA12 = isClose(step5Inputs.a12, ref.A[0][1], 0.01);

    const isSuccess = okQkt11 && okQkt12 && okS11 && okS12 && okA11 && okA12;
    recordValidation(5, isSuccess);

    setStep5Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? `Impressionnant ! QKᵀ[1,:]=[${ref.QKT[0].join(', ')}], S[1,:]=[${ref.S[0].map(v=>v.toFixed(3)).join(', ')}], A[1,:]=[${ref.A[0].map(v=>v.toFixed(3)).join(', ')}].`
        : `QKᵀ = Q·Kᵀ : ligne 1 = [2×1+0×1+1×0+0×0, 2×0+0×2+1×1+0×0] = [${ref.QKT[0].join(', ')}]. S = QKᵀ/√4 = [${ref.S[0].map(v=>v.toFixed(3)).join(', ')}]. Softmax → A=[${ref.A[0].map(v=>v.toFixed(3)).join(', ')}].`,
    });
  };

  const verifyStep6 = () => {
    // Facteur expansion = 4, D_MLP = 4×768=3072 (réel) ou 4×4=16 (mini)
    const expectedMlpDim = isReal ? 3072 : 16;
    const okExp = isClose(step6Inputs.expansionFactor, 4, 0.1);
    const okMlp = isClose(step6Inputs.mlpDim, expectedMlpDim, isReal ? 10 : 1);
    const okGelu1 = isClose(step6Inputs.geluZ1, 2.996, 0.01) || isClose(step6Inputs.geluZ1, 3.0, 0.01);
    const okGelu2 = isClose(step6Inputs.geluZ2, -0.159, 0.01) || isClose(step6Inputs.geluZ2, -0.158, 0.01);
    const isSuccess = okExp && okMlp && okGelu1 && okGelu2;
    recordValidation(6, isSuccess);

    setStep6Feedback({
      isChecked: true,
      isCorrect: isSuccess,
      message: isSuccess
        ? `Bravo ! Facteur=4, D_MLP=4×${isReal ? 768 : 4}=${expectedMlpDim}. GELU(3.0)≈2.996, GELU(−1.0)≈−0.159.`
        : `Facteur=4, D_MLP=4×${isReal ? 768 : 4}=${expectedMlpDim}. GELU(3.0)=0.5×3×(1+tanh(√(2/π)×(3+0.044715×27)))≈2.996. GELU(−1.0)≈−0.159.`,
    });
  };

  const verifyStep7 = () => {
    const ref = step7Ref;
    // ViT: z_cls=[2,1,0], exp(2)=7.389, sum=11.107, P1=0.665
    const okExp1 = isClose(step7Inputs.exp1, ref.softmaxCls.exps[0], 0.01); // 7.389
    const okSum = isClose(step7Inputs.sumExp, ref.softmaxCls.sumExp, 0.01);  // 11.107
    const okP1 = isClose(step7Inputs.prob1, ref.softmaxCls.probs[0], 0.01); // 0.665

    if (selectedModel === 'vit') {
      const isSuccess = okExp1 && okSum && okP1;
      recordValidation(7, isSuccess);
      setStep7Feedback({
        isChecked: true,
        isCorrect: isSuccess,
        message: isSuccess
          ? 'Bravo ! exp(2.0)=7.389, somme=11.107, P(classe 1)=7.389/11.107≈0.665. Classe prédite = 1.'
          : `exp(2.0)=e²≈${ref.softmaxCls.exps[0].toFixed(3)}, somme=e²+e¹+e⁰≈${ref.softmaxCls.sumExp.toFixed(3)}, P₁=7.389/11.107≈${ref.softmaxCls.probs[0].toFixed(3)}.`,
      });
    } else {
      // DeiT: also check paper fusion
      const okPf1 = isClose(step7Inputs.pFinalPaper1, ref.pFinalPaper[0], 0.01);
      const okPf2 = isClose(step7Inputs.pFinalPaper2, ref.pFinalPaper[1], 0.01);
      const okPf3 = isClose(step7Inputs.pFinalPaper3, ref.pFinalPaper[2], 0.01);
      const isSuccess = okExp1 && okSum && okP1 && okPf1 && okPf2 && okPf3;
      recordValidation(7, isSuccess);
      setStep7Feedback({
        isChecked: true,
        isCorrect: isSuccess,
        message: isSuccess
          ? `Félicitations ! Fusion papier : P_final=[${ref.pFinalPaper.map(v=>v.toFixed(3)).join(', ')}] → classe ${ref.winnerPaper} gagne (les têtes étaient en désaccord, la fusion tranche).`
          : `P_cls≈[${ref.softmaxCls.probs.map(v=>v.toFixed(3)).join(', ')}], P_dist≈[${ref.softmaxDist.probs.map(v=>v.toFixed(3)).join(', ')}]. Fusion : ½(P_cls+P_dist)=[${ref.pFinalPaper.map(v=>v.toFixed(3)).join(', ')}] → classe ${ref.winnerPaper}.`,
      });
    }
  };

  const completedCount = Object.keys(validatedTheorySteps).length;
  const progressPct = Math.round((completedCount / 7) * 100);

  // ─────────────────────────────────────────────────────────────────────────
  // RENDER
  // ─────────────────────────────────────────────────────────────────────────
  return (
    <div className="unified-page-container">
      {/* ── HERO BANNER ── */}
      <div className="unified-banner-card">
        <div className="banner-left-content">
          <div className="banner-icon-box">
            <Calculator size={28} className="banner-main-icon" />
          </div>
          <div className="banner-titles">
            <h1 className="banner-title">Laboratoire théorique : de l'image à la classification</h1>
            <p className="banner-subtitle">
              Réalisez vous-même les calculs mathématiques d'un Vision Transformer et d'un DeiT • Tolérance ±0,01
            </p>
          </div>
        </div>

        {/* Global Selectors */}
        <div className="banner-controls-group">
          {/* Model Toggle */}
          <div className="control-pill-toggle">
            <span className="pill-toggle-label">Architecture :</span>
            <button
              type="button"
              className={`pill-toggle-btn vit-btn ${selectedModel === 'vit' ? 'active' : ''}`}
              onClick={() => setSelectedModel('vit')}
              title="Architecture Google ViT — 197 tokens"
            >
              <Brain size={15} />
              <span>ViT</span>
              {selectedModel === 'vit' && <span className="active-dot vit" />}
            </button>
            <button
              type="button"
              className={`pill-toggle-btn deit-btn ${selectedModel === 'deit' ? 'active' : ''}`}
              onClick={() => setSelectedModel('deit')}
              title="Architecture Meta DeiT — 198 tokens + distillation"
            >
              <Cpu size={15} />
              <span>DeiT</span>
              {selectedModel === 'deit' && <span className="active-dot deit" />}
            </button>
          </div>

          {/* Mode Toggle */}
          <div className="control-pill-toggle">
            <span className="pill-toggle-label">Mode :</span>
            <button
              type="button"
              className={`pill-toggle-btn scale-btn ${diffLevel === 'miniature' ? 'active' : ''}`}
              onClick={() => setDiffLevel('miniature')}
              title="6×6, P=2, N=9, D_model=4 — calcul mental rapide"
            >
              <span>Miniature</span>
            </button>
            <button
              type="button"
              className={`pill-toggle-btn scale-btn ${diffLevel === 'real' ? 'active' : ''}`}
              onClick={() => setDiffLevel('real')}
              title="224×224, P=16, N=196, D=768 — dimensions réelles ViT-Base"
            >
              <span>Réel</span>
            </button>
          </div>

          {/* Status chip */}
          <div className={`active-model-status-chip ${selectedModel}`}>
            <span className="chip-badge">
              {selectedModel === 'deit' ? 'DeiT' : 'ViT'} — {diffLevel === 'miniature' ? 'Miniature' : 'Réel'}
            </span>
            <span className="chip-desc">
              {diffLevel === 'miniature'
                ? `6×6, P=2, N=9, D=${selectedModel === 'vit' ? 10 : 11} tokens`
                : `224×224, P=${step1PatchSize}, N=${patchStats.totalPatches}, ${seqLenExpected} tokens`}
            </span>
          </div>
        </div>
      </div>

      {/* ── Miniature / Réel INFO BAR ── */}
      <div className="mode-info-banner" style={{
        background: diffLevel === 'miniature' ? '#fefce8' : '#eff6ff',
        border: `1px solid ${diffLevel === 'miniature' ? '#fde68a' : '#bfdbfe'}`,
        borderRadius: '10px',
        padding: '10px 18px',
        display: 'flex',
        alignItems: 'center',
        gap: '12px',
        fontSize: '12.5px',
        color: diffLevel === 'miniature' ? '#78350f' : '#1e40af',
      }}>
        <Info size={16} />
        {diffLevel === 'miniature' ? (
          <span>
            <strong>Mode Miniature :</strong> Image 6×6, P=2, C=3 → <strong>9 patches</strong>, D_in=12, D_model=4.
            Séquence ViT = <strong>10 tokens</strong>, DeiT = <strong>11 tokens</strong>. Idéal pour le calcul mental.
          </span>
        ) : (
          <span>
            <strong>Mode Réel :</strong> Image 224×224, P={step1PatchSize}, C=3 → <strong>{patchStats.totalPatches} patches</strong>, D_in={patchStats.flattenedDim}, D_model=768.
            Séquence ViT = <strong>{getSeqLen(patchStats.totalPatches, 'vit')} tokens</strong>, DeiT = <strong>{getSeqLen(patchStats.totalPatches, 'deit')} tokens</strong>. Dimensions réelles ViT-Base.
          </span>
        )}
      </div>

      {/* ── PROGRESS BAR ── */}
      <div className="lab-progress-bar-card">
        <div className="lab-stats-left">
          <div className="lab-stat-item">
            <span className="stat-label">Progression :</span>
            <div className="stat-progress-pill">
              <div className="stat-track">
                <div className="stat-fill" style={{ width: `${progressPct}%` }} />
              </div>
              <span className="stat-pct-num">{completedCount} / 7 étapes ({progressPct}%)</span>
            </div>
          </div>
          <div className="lab-stat-item">
            <span className="stat-label">Score :</span>
            <span className="score-badge"><Award size={14} /><b>{score} / 100 pts</b></span>
          </div>
          <div className="lab-stat-item">
            <span className="stat-label">Tentatives :</span>
            <span className="attempts-badge">{attempts}</span>
          </div>
        </div>
        <button type="button" className="btn-reset-lab" onClick={() => { if (window.confirm('Réinitialiser ?')) resetAllProgress(); }}>
          <RotateCcw size={14} /><span>Réinitialiser</span>
        </button>
      </div>

      {/* ── STEPPER ── */}
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

      {/* ── EXERCISE CARD ── */}
      <div className="lab-exercise-card">

        {/* ══════════════════════════════════════════════════════════════════
            ÉTAPE 1 — DÉCOUPAGE EN PATCHES (commune ViT & DeiT)
        ══════════════════════════════════════════════════════════════════ */}
        {currentStep === 1 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 01 / 07 — commune à ViT et DeiT</span>
              <h3>Découpage d'une image en patches et calcul des dimensions</h3>
              <p>
                Une image <InlineMath math="\mathbf{X} \in \mathbb{R}^{H \times W \times C}" /> est découpée
                en une grille régulière de patches carrés de côté <InlineMath math="P" />.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">1</span>
                  <span className="panel-header-title">Formules & Exercice</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="N = \frac{H \times W}{P^2} = \frac{H}{P} \times \frac{W}{P}, \qquad D_{\text{flat}} = P^2 \times C" />
                </div>

                {/* Patch size selector — only in real mode */}
                {isReal && (
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
                          {p}×{p}
                        </button>
                      ))}
                    </div>
                  </div>
                )}

                {/* Données */}
                <div className="context-data-box">
                  {isReal ? (
                    <span>Données : <InlineMath math={`H=${224},\\; W=${224},\\; P=${step1PatchSize},\\; C=3`} /></span>
                  ) : (
                    <span>Données (Miniature) : <InlineMath math="H=6,\; W=6,\; P=2,\; C=3" /></span>
                  )}
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card">
                    <label>1. Nombre de patches par côté (<InlineMath math="H/P" />) :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step1Inputs.gridRowsCols}
                      onChange={(e) => setStep1Inputs({ ...step1Inputs, gridRowsCols: e.target.value })}
                      placeholder="Entrez un nombre"
                    />
                  </div>
                  <div className="input-field-card">
                    <label>2. Nombre total de patches (<InlineMath math="N" />) :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step1Inputs.totalPatches}
                      onChange={(e) => setStep1Inputs({ ...step1Inputs, totalPatches: e.target.value })}
                      placeholder="Entrez un nombre"
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
                        placeholder="Entrez un nombre"
                      />
                      <span className="unit-label">valeurs</span>
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep1}>
                    <CheckCircle2 size={16} /><span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-hint" onClick={() => showNextHint(1)}>
                    <HelpCircle size={15} /><span>Indice {hintLevels[1] ? `(${hintLevels[1]}/3)` : ''}</span>
                  </button>
                  <button type="button" className="btn-toggle-solution" onClick={() => setSolutionVisible({ ...solutionVisible, 1: !solutionVisible[1] })}>
                    <span>{solutionVisible[1] ? 'Masquer la correction' : 'Voir la correction détaillée'}</span>
                  </button>
                </div>

                {step1Feedback && (
                  <div className={`feedback-alert-box ${step1Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step1Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step1Feedback.message}</span>
                  </div>
                )}

                {hintLevels[1] >= 1 && (
                  <div className="hint-collapsible-box">
                    <strong>Indice {hintLevels[1]} :</strong>{' '}
                    {hintLevels[1] === 1 && `Divisez chaque côté (${isReal ? 224 : 6} px) par la taille du patch (${isReal ? step1PatchSize : 2} px).`}
                    {hintLevels[1] === 2 && `Multipliez le résultat par lui-même pour obtenir N (grille carrée).`}
                    {hintLevels[1] >= 3 && `D_flat = P² × C. Chaque pixel a 3 valeurs (R, G, B).`}
                  </div>
                )}

                {solutionVisible[1] && (
                  <div className="solution-collapsible-box">
                    <h4>Correction détaillée :</h4>
                    {isReal ? (
                      <>
                        <p>1. Patches par côté : <InlineMath math={`\\frac{224}{${step1PatchSize}} = ${patchStats.gridRows}`} /></p>
                        <p>2. Total : <InlineMath math={`N = ${patchStats.gridRows}^2 = ${patchStats.totalPatches}`} /></p>
                        <p>3. Vecteur aplati : <InlineMath math={`D_{\\text{flat}} = ${step1PatchSize}^2 \\times 3 = ${patchStats.flattenedDim}`} /></p>
                      </>
                    ) : (
                      <>
                        <p>1. Patches par côté : <InlineMath math="\frac{6}{2} = 3" /></p>
                        <p>2. Total : <InlineMath math="N = 3^2 = 9" /></p>
                        <p>3. Vecteur aplati : <InlineMath math="D_{\text{flat}} = 2^2 \times 3 = 12" /></p>
                      </>
                    )}
                  </div>
                )}
              </div>

              {/* Visual panel */}
              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Grille de patches dynamique</span>
                </div>
                {isReal ? (
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
                      Grille : <b>{224 / step1PatchSize}×{224 / step1PatchSize}</b> ({' '}
                      <b>{(224 / step1PatchSize) ** 2}</b> patches de {step1PatchSize}×{step1PatchSize} px)
                    </span>
                  </div>
                ) : (
                  <div className="dynamic-grid-viewport" style={{ padding: '14px' }}>
                    <div style={{
                      display: 'grid',
                      gridTemplateColumns: 'repeat(3, 1fr)',
                      gap: '4px',
                      width: '90px',
                      margin: '0 auto 8px',
                    }}>
                      {Array.from({ length: 9 }).map((_, i) => (
                        <div key={i} style={{
                          width: '28px', height: '28px',
                          borderRadius: '4px',
                          border: '1px solid #93c5fd',
                          background: `hsl(${200 + i * 15}, 60%, 80%)`,
                          display: 'flex', alignItems: 'center', justifyContent: 'center',
                          fontSize: '10px', fontWeight: '700', color: '#1e3a8a',
                        }}>{i + 1}</div>
                      ))}
                    </div>
                    <span className="grid-meta-summary">Miniature 6×6 → 9 patches de 2×2 px</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            ÉTAPE 2 — PROJECTION LINÉAIRE (commune ViT & DeiT)
        ══════════════════════════════════════════════════════════════════ */}
        {currentStep === 2 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 02 / 07 — commune à ViT et DeiT</span>
              <h3>Projection linéaire : du patch aplati à l'embedding</h3>
              <p>
                Chaque patch aplati <InlineMath math="\mathbf{x}_p \in \mathbb{R}^{D_{\text{in}}}" /> est projeté
                via la matrice apprenante <InlineMath math="\mathbf{E} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{model}}}" />.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">2</span>
                  <span className="panel-header-title">Dimensions & Multiplication Matricielle</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="\mathbf{z}_p = \mathbf{x}_p\,\mathbf{E} + \mathbf{b}, \quad \mathbf{E} \in \mathbb{R}^{D_{\text{in}} \times D_{\text{model}}}" />
                </div>

                <div className="info-banner-blue" style={{ marginBottom: '12px' }}>
                  <Info size={15} />
                  <span>
                    <strong>Note implémentation :</strong> Le papier ViT original n'inclut pas de biais.
                    HuggingFace utilise une <code>Conv2d(C, D_model, P, stride=P)</code> avec biais.
                    Dans les formules, <strong>E désigne uniquement la matrice de projection</strong> (jamais un embedding de token).
                    Les vecteurs de patch après projection sont notés <strong>z_p</strong>.
                  </span>
                </div>

                <div className="context-data-box">
                  {isReal
                    ? <span>Mode Réel : D_in = D_flat = {patchStats.flattenedDim}, D_model = 768</span>
                    : <span>Mode Miniature : D_in = 12 (= 2²×3), D_model = 4</span>}
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card">
                    <label>1. Dimension d'entrée <InlineMath math="D_{\text{in}}" /> :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step2Inputs.dimIn}
                      onChange={(e) => setStep2Inputs({ ...step2Inputs, dimIn: e.target.value })}
                      placeholder="Entrez un nombre"
                    />
                  </div>

                  <div className="input-field-card">
                    <label>2. Dimension de sortie <InlineMath math="D_{\text{model}}" /> :</label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step2Inputs.dimOut}
                      onChange={(e) => setStep2Inputs({ ...step2Inputs, dimOut: e.target.value })}
                      placeholder="Entrez un nombre"
                    />
                  </div>

                  <div className="sub-calc-step">
                    <span className="grp-label">3. Multiplication manuelle (exemple numérique) :</span>
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
                    <CheckCircle2 size={16} /><span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-hint" onClick={() => showNextHint(2)}>
                    <HelpCircle size={15} /><span>Indice {hintLevels[2] ? `(${hintLevels[2]}/3)` : ''}</span>
                  </button>
                  <button type="button" className="btn-toggle-solution" onClick={() => setSolutionVisible({ ...solutionVisible, 2: !solutionVisible[2] })}>
                    <span>{solutionVisible[2] ? 'Masquer' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step2Feedback && (
                  <div className={`feedback-alert-box ${step2Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step2Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step2Feedback.message}</span>
                  </div>
                )}

                {hintLevels[2] >= 1 && (
                  <div className="hint-collapsible-box">
                    <strong>Indice {hintLevels[2]} :</strong>{' '}
                    {hintLevels[2] === 1 && 'D_in = taille du patch aplati = P² × C.'}
                    {hintLevels[2] === 2 && 'z₁ = somme de x_i × W_i1 pour i=1..3, puis + b₁.'}
                    {hintLevels[2] >= 3 && 'z₁ = 2×1 + 1×0 + 3×1 + 1 = 6. z₂ = 2×0 + 1×2 + 3×1 − 1 = 4.'}
                  </div>
                )}

                {solutionVisible[2] && (
                  <div className="solution-collapsible-box">
                    <h4>Correction :</h4>
                    <p>D_in = {isReal ? patchStats.flattenedDim : 12} (= {isReal ? step1PatchSize : 2}²×3), D_model = {isReal ? 768 : 4}.</p>
                    <p><InlineMath math="z_1 = 2(1) + 1(0) + 3(1) + 1 = 6" /></p>
                    <p><InlineMath math="z_2 = 2(0) + 1(2) + 3(1) - 1 = 4" /></p>
                  </div>
                )}
              </div>

              {/* Visual: Embedding ribbon */}
              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Ruban des vecteurs z_p après projection</span>
                </div>
                <div style={{ padding: '12px 10px' }}>
                  <EmbeddingRibbon
                    N={isReal ? patchStats.totalPatches : 9}
                    dModel={isReal ? 768 : 4}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            ÉTAPE 3 — TOKENS & ENCODAGE POSITIONNEL
        ══════════════════════════════════════════════════════════════════ */}
        {currentStep === 3 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">
                ÉTAPE 03 / 07 — {selectedModel === 'vit' ? 'ViT : N+1 tokens' : 'DeiT : N+2 tokens'}
              </span>
              <h3>Construction des tokens et somme d'encodage positionnel</h3>
              <p>
                ViT prépose un token <code>[CLS]</code>. DeiT ajoute également un token <code>[DIST]</code>.
                On additionne ensuite l'encodage positionnel appris <InlineMath math="\mathbf{E}_{\text{pos}}" />.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">3</span>
                  <span className="panel-header-title">Séquence & Addition Positionnelle</span>
                </div>

                <div className="formula-callout-box">
                  {selectedModel === 'deit' ? (
                    <BlockMath math="\mathbf{Z}_{\text{DeiT}} = \bigl[\mathbf{x}_{\text{cls}};\; \mathbf{x}_{\text{dist}};\; \mathbf{z}_p^1;\; \dots;\; \mathbf{z}_p^N\bigr] + \mathbf{E}_{\text{pos}} \quad (N+2 \text{ tokens})" />
                  ) : (
                    <BlockMath math="\mathbf{Z}_{\text{ViT}} = \bigl[\mathbf{x}_{\text{cls}};\; \mathbf{z}_p^1;\; \dots;\; \mathbf{z}_p^N\bigr] + \mathbf{E}_{\text{pos}} \quad (N+1 \text{ tokens})" />
                  )}
                </div>

                <div className="info-banner-blue" style={{ marginBottom: '10px', fontSize: '11.5px' }}>
                  <Info size={14} />
                  <span>
                    <strong>[CLS]</strong> et <strong>[DIST]</strong> ont chacun leur propre embedding appris et leur propre ligne dans <InlineMath math="\mathbf{E}_{\text{pos}}" />.
                    {selectedModel === 'vit'
                      ? ` ViT-Base : E_pos ∈ ℝ^{197×768} (1 spécial + 196 patches).`
                      : ` DeiT-Base : E_pos ∈ ℝ^{198×768} (2 spéciaux + 196 patches).`}
                  </span>
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card highlighted-active-model">
                    <label>
                      1. Longueur totale de séquence pour {selectedModel === 'vit' ? 'ViT' : 'DeiT'}{' '}
                      ({isReal ? patchStats.totalPatches : 9} patches) :
                      <span className={`active-model-tag ${selectedModel === 'deit' ? 'deit' : ''}`}>Actif</span>
                    </label>
                    <input
                      type="number"
                      className="styled-lab-input"
                      value={step3Inputs.seqLen}
                      onChange={(e) => setStep3Inputs({ ...step3Inputs, seqLen: e.target.value })}
                      placeholder="Entrez un nombre"
                    />
                  </div>

                  <div className="sub-calc-step">
                    <span className="grp-label">2. Addition vectorielle d'encodage positionnel :</span>
                    <p className="desc-text">
                      Soit <InlineMath math="\mathbf{z}_p = [0.5,\; -0.2,\; 1.0]" /> et{' '}
                      <InlineMath math="\mathbf{e}_{\text{pos}} = [0.1,\; 0.4,\; -0.3]" />. Calculez <InlineMath math="\mathbf{Z} = \mathbf{z}_p + \mathbf{e}_{\text{pos}}" /> :
                    </p>
                    <div className="result-cells-row">
                      {['Z₁', 'Z₂', 'Z₃'].map((label, i) => (
                        <div className="vec-box" key={i}>
                          <span>{label} = </span>
                          <input
                            type="number"
                            step="0.1"
                            className="cell-input"
                            value={[step3Inputs.zPos1, step3Inputs.zPos2, step3Inputs.zPos3][i]}
                            onChange={(e) => {
                              const keys = ['zPos1', 'zPos2', 'zPos3'] as const;
                              setStep3Inputs({ ...step3Inputs, [keys[i]]: e.target.value });
                            }}
                            placeholder="?"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep3}>
                    <CheckCircle2 size={16} /><span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-solution" onClick={() => setSolutionVisible({ ...solutionVisible, 3: !solutionVisible[3] })}>
                    <span>{solutionVisible[3] ? 'Masquer' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step3Feedback && (
                  <div className={`feedback-alert-box ${step3Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step3Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step3Feedback.message}</span>
                  </div>
                )}

                {solutionVisible[3] && (
                  <div className="solution-collapsible-box">
                    <h4>Correction :</h4>
                    <p>N = {isReal ? patchStats.totalPatches : 9} patches</p>
                    <p>{selectedModel === 'vit'
                      ? `ViT : N + 1 = ${seqLenExpected} tokens ([CLS] + ${isReal ? patchStats.totalPatches : 9} patches).`
                      : `DeiT : N + 2 = ${seqLenExpected} tokens ([CLS] + [DIST] + ${isReal ? patchStats.totalPatches : 9} patches).`}</p>
                    <p><InlineMath math="\mathbf{Z} = [0.5+0.1,\; -0.2+0.4,\; 1.0-0.3] = [0.6,\; 0.2,\; 0.7]" /></p>
                  </div>
                )}
              </div>

              {/* Visual: Token Sequence */}
              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Séquence de tokens (interactif)</span>
                </div>
                <div style={{ padding: '12px 10px' }}>
                  <TokenSequenceVisualizer
                    model={selectedModel as 'vit' | 'deit'}
                    N={isReal ? patchStats.totalPatches : 9}
                    dModel={isReal ? 768 : 4}
                  />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            ÉTAPE 4 — LAYER NORMALIZATION (commune ViT & DeiT)
        ══════════════════════════════════════════════════════════════════ */}
        {currentStep === 4 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 04 / 07 — commune à ViT et DeiT</span>
              <h3>Layer Normalization — centrage et réduction d'un token</h3>
              <p>
                Chaque token est normalisé individuellement selon la formule complète avec paramètres appris γ et β.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">4</span>
                  <span className="panel-header-title">Vecteur exemple : [1, 3, 3, 9]</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="\text{LN}(\mathbf{x}) = \gamma \odot \hat{\mathbf{x}} + \beta, \quad \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \varepsilon}}" />
                  <BlockMath math="\mu = \frac{1}{D}\sum_{i=1}^D x_i, \qquad \sigma^2 = \frac{1}{D}\sum_{i=1}^D (x_i - \mu)^2" />
                </div>

                <div className="info-banner-blue" style={{ marginBottom: '10px', fontSize: '11.5px' }}>
                  <Info size={14} />
                  <span>ε = 10⁻⁵ (stabilité numérique). Dans cet exercice, γ=1 et β=0 (initialisation). Calculez <InlineMath math="\hat{\mathbf{x}}" />, pas la sortie finale LN.</span>
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card">
                    <label>1. Moyenne <InlineMath math="\mu" /> :</label>
                    <input type="number" step="0.1" className="styled-lab-input"
                      value={step4Inputs.mean}
                      onChange={(e) => setStep4Inputs({ ...step4Inputs, mean: e.target.value })}
                      placeholder="Entrez un nombre" />
                  </div>
                  <div className="input-field-card">
                    <label>2. Variance <InlineMath math="\sigma^2" /> :</label>
                    <input type="number" step="0.1" className="styled-lab-input"
                      value={step4Inputs.variance}
                      onChange={(e) => setStep4Inputs({ ...step4Inputs, variance: e.target.value })}
                      placeholder="Entrez un nombre" />
                  </div>
                  <div className="input-field-card">
                    <label>3. Écart-type <InlineMath math="\sigma = \sqrt{\sigma^2}" /> :</label>
                    <input type="number" step="0.1" className="styled-lab-input"
                      value={step4Inputs.stdDev}
                      onChange={(e) => setStep4Inputs({ ...step4Inputs, stdDev: e.target.value })}
                      placeholder="Entrez un nombre" />
                  </div>
                  <div className="sub-calc-step">
                    <span className="grp-label">4. Vecteur normalisé <InlineMath math="\hat{\mathbf{x}}" /> (3 décimales) :</span>
                    <div className="result-cells-row">
                      {['x̂₁', 'x̂₂', 'x̂₃', 'x̂₄'].map((label, i) => {
                        const keys = ['xNorm1', 'xNorm2', 'xNorm3', 'xNorm4'] as const;
                        return (
                          <input
                            key={i}
                            type="number"
                            step="0.001"
                            className="cell-input"
                            value={step4Inputs[keys[i]]}
                            onChange={(e) => setStep4Inputs({ ...step4Inputs, [keys[i]]: e.target.value })}
                            placeholder={label}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep4}>
                    <CheckCircle2 size={16} /><span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-hint" onClick={() => showNextHint(4)}>
                    <HelpCircle size={15} /><span>Indice {hintLevels[4] ? `(${hintLevels[4]}/3)` : ''}</span>
                  </button>
                  <button type="button" className="btn-toggle-solution" onClick={() => setSolutionVisible({ ...solutionVisible, 4: !solutionVisible[4] })}>
                    <span>{solutionVisible[4] ? 'Masquer' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step4Feedback && (
                  <div className={`feedback-alert-box ${step4Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step4Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step4Feedback.message}</span>
                  </div>
                )}

                {hintLevels[4] >= 1 && (
                  <div className="hint-collapsible-box">
                    <strong>Indice {hintLevels[4]} :</strong>{' '}
                    {hintLevels[4] === 1 && 'μ = somme des éléments divisée par 4.'}
                    {hintLevels[4] === 2 && 'σ² = moyenne des (xᵢ − μ)². Pour chaque terme : (1−4)²=9, (3−4)²=1, (3−4)²=1, (9−4)²=25.'}
                    {hintLevels[4] >= 3 && 'μ=4, σ²=36/4=9, σ=3. x̂₁=(1−4)/3=−1.000, x̂₂=(3−4)/3=−0.333, x̂₃=−0.333, x̂₄=(9−4)/3=1.667.'}
                  </div>
                )}

                {solutionVisible[4] && (
                  <div className="solution-collapsible-box">
                    <h4>Correction détaillée :</h4>
                    <p><InlineMath math="\mu = \frac{1+3+3+9}{4} = \frac{16}{4} = 4.000" /></p>
                    <p><InlineMath math="\sigma^2 = \frac{(1-4)^2+(3-4)^2+(3-4)^2+(9-4)^2}{4} = \frac{9+1+1+25}{4} = 9.000" /></p>
                    <p><InlineMath math="\sigma = \sqrt{9} = 3.000" /></p>
                    <p><InlineMath math="\hat{x}_1 = \frac{1-4}{3} = -1.000, \quad \hat{x}_2 = \frac{3-4}{3} = -0.333" /></p>
                    <p><InlineMath math="\hat{x}_3 = -0.333, \quad \hat{x}_4 = \frac{9-4}{3} = 1.667" /></p>
                  </div>
                )}
              </div>

              {/* Visual: bar chart brut vs normalisé */}
              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Centrage & Réduction (μ=4, σ=3)</span>
                </div>
                <div className="norm-comparison-bars" style={{ padding: '14px', gap: '20px' }}>
                  {/* Raw values */}
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '11px', fontWeight: '700', color: '#64748b', marginBottom: '6px', textAlign: 'center' }}>
                      Brut [1, 3, 3, 9]
                    </div>
                    <div style={{ display: 'flex', alignItems: 'flex-end', gap: '6px', height: '100px', justifyContent: 'center' }}>
                      {[1, 3, 3, 9].map((v, i) => (
                        <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
                          <span style={{ fontSize: '9px', color: '#64748b' }}>{v}</span>
                          <div style={{ width: '22px', height: `${(v / 9) * 80}px`, background: '#93c5fd', borderRadius: '3px 3px 0 0' }} />
                          <span style={{ fontSize: '9px', color: '#94a3b8' }}>x{i+1}</span>
                        </div>
                      ))}
                    </div>
                    <div style={{ fontSize: '10px', color: '#94a3b8', textAlign: 'center', marginTop: '4px' }}>μ=4, σ=3</div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', fontSize: '20px', color: '#6366f1' }}>→</div>

                  {/* Normalized values */}
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: '11px', fontWeight: '700', color: '#64748b', marginBottom: '6px', textAlign: 'center' }}>
                      Normalisé x̂
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '6px', height: '100px', justifyContent: 'center' }}>
                      {[-1.0, -0.333, -0.333, 1.667].map((v, i) => {
                        const absV = Math.abs(v);
                        const isNeg = v < 0;
                        return (
                          <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
                            {!isNeg && <span style={{ fontSize: '8px', color: '#065f46' }}>{v.toFixed(3)}</span>}
                            {!isNeg && <div style={{ width: '22px', height: `${absV * 35}px`, background: '#34d399', borderRadius: '3px 3px 0 0' }} />}
                            <div style={{ width: '1px', height: '2px', background: '#94a3b8' }} />
                            {isNeg && <div style={{ width: '22px', height: `${absV * 35}px`, background: '#fca5a5', borderRadius: '0 0 3px 3px' }} />}
                            {isNeg && <span style={{ fontSize: '8px', color: '#dc2626' }}>{v.toFixed(3)}</span>}
                            <span style={{ fontSize: '9px', color: '#94a3b8' }}>x̂{i+1}</span>
                          </div>
                        );
                      })}
                    </div>
                    <div style={{ fontSize: '10px', color: '#94a3b8', textAlign: 'center', marginTop: '4px' }}>μ=0, σ=1</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            ÉTAPE 5 — ATTENTION MULTI-HEAD (commune ViT & DeiT)
        ══════════════════════════════════════════════════════════════════ */}
        {currentStep === 5 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">
                ÉTAPE 05 / 07 — commune (matrice {selectedModel === 'deit' ? '198×198' : '197×197'} en pratique)
              </span>
              <h3>Scaled Dot-Product Attention — calcul de QKᵀ, S et Softmax</h3>
              <p>
                On donne <InlineMath math="\mathbf{Q} = \begin{pmatrix}2&0&1&0\\1&1&0&1\end{pmatrix}" />,{' '}
                <InlineMath math="\mathbf{K} = \begin{pmatrix}1&1&0&0\\0&2&1&0\end{pmatrix}" />,{' '}
                <InlineMath math="d_k = 4" /> (donc <InlineMath math="\sqrt{d_k} = 2" />).
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">5</span>
                  <span className="panel-header-title">Scaled Dot-Product Attention</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}, \qquad \mathbf{A} = \text{softmax}(\mathbf{S}) \quad \text{(ligne par ligne)}" />
                </div>

                {selectedModel === 'deit' && (
                  <div className="info-banner-blue" style={{ marginBottom: '10px', fontSize: '11.5px' }}>
                    <Info size={14} />
                    <span>Pour DeiT, la matrice d'attention complète est <strong>198×198</strong> (un token [DIST] supplémentaire). Le calcul reste identique.</span>
                  </div>
                )}

                <div className="exercise-inputs-grid">
                  {/* QKT row 1 */}
                  <div className="sub-calc-step">
                    <span className="grp-label">1. <InlineMath math="\mathbf{Q}\mathbf{K}^T" /> — première ligne (tokens 1 et 2) :</span>
                    <p className="desc-text" style={{ fontSize: '11px' }}>
                      Rappel : Q[0]=[2,0,1,0], K[0]=[1,1,0,0], K[1]=[0,2,1,0].
                    </p>
                    <div className="result-cells-row">
                      <span>QKᵀ[1,1] = </span>
                      <input type="number" className="cell-input"
                        value={step5Inputs.qkt11}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, qkt11: e.target.value })}
                        placeholder="?" />
                      <span>QKᵀ[1,2] = </span>
                      <input type="number" className="cell-input"
                        value={step5Inputs.qkt12}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, qkt12: e.target.value })}
                        placeholder="?" />
                    </div>
                  </div>

                  {/* Scores S row 1 */}
                  <div className="sub-calc-step">
                    <span className="grp-label">2. Scores échelonnés <InlineMath math="S = QK^T / \sqrt{4}" /> — ligne 1 :</span>
                    <div className="result-cells-row">
                      <span>S[1,1] = </span>
                      <input type="number" step="0.1" className="cell-input"
                        value={step5Inputs.s11}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, s11: e.target.value })}
                        placeholder="?" />
                      <span>S[1,2] = </span>
                      <input type="number" step="0.1" className="cell-input"
                        value={step5Inputs.s12}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, s12: e.target.value })}
                        placeholder="?" />
                    </div>
                  </div>

                  {/* Softmax A row 1 */}
                  <div className="sub-calc-step">
                    <span className="grp-label">3. Poids d'attention Softmax — ligne 1 :</span>
                    <div className="result-cells-row">
                      <span>A[1,1] = </span>
                      <input type="number" step="0.001" className="cell-input"
                        value={step5Inputs.a11}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, a11: e.target.value })}
                        placeholder="?" />
                      <span>A[1,2] = </span>
                      <input type="number" step="0.001" className="cell-input"
                        value={step5Inputs.a12}
                        onChange={(e) => setStep5Inputs({ ...step5Inputs, a12: e.target.value })}
                        placeholder="?" />
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep5}>
                    <CheckCircle2 size={16} /><span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-hint" onClick={() => showNextHint(5)}>
                    <HelpCircle size={15} /><span>Indice {hintLevels[5] ? `(${hintLevels[5]}/3)` : ''}</span>
                  </button>
                  <button type="button" className="btn-toggle-solution" onClick={() => setSolutionVisible({ ...solutionVisible, 5: !solutionVisible[5] })}>
                    <span>{solutionVisible[5] ? 'Masquer' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step5Feedback && (
                  <div className={`feedback-alert-box ${step5Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step5Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step5Feedback.message}</span>
                  </div>
                )}

                {hintLevels[5] >= 1 && (
                  <div className="hint-collapsible-box">
                    <strong>Indice {hintLevels[5]} :</strong>{' '}
                    {hintLevels[5] === 1 && 'QKᵀ[1,j] = somme de Q[1,k] × K[j,k] pour k=0..3 (produit scalaire ligne×ligne).'}
                    {hintLevels[5] === 2 && `QKᵀ[1,1] = 2×1+0×1+1×0+0×0 = ${attnRef.QKT[0][0]}. QKᵀ[1,2] = 2×0+0×2+1×1+0×0 = ${attnRef.QKT[0][1]}.`}
                    {hintLevels[5] >= 3 && `S[1,:]=[${attnRef.S[0].map(v=>v.toFixed(3)).join(', ')}]. Softmax(S[1,:]) = [e^${attnRef.S[0][0]}/(e^${attnRef.S[0][0]}+e^${attnRef.S[0][1]}), ...] ≈ [${attnRef.A[0].map(v=>v.toFixed(3)).join(', ')}].`}
                  </div>
                )}

                {solutionVisible[5] && (
                  <div className="solution-collapsible-box">
                    <h4>Correction :</h4>
                    <p><InlineMath math={`\\mathbf{QK}^T = \\begin{pmatrix}${attnRef.QKT[0].join('&')}\\\\${attnRef.QKT[1].join('&')}\\end{pmatrix}`} /></p>
                    <p><InlineMath math={`\\mathbf{S} = \\begin{pmatrix}${attnRef.S[0].map(v=>v.toFixed(3)).join('&')}\\\\${attnRef.S[1].map(v=>v.toFixed(3)).join('&')}\\end{pmatrix}`} /></p>
                    <p><InlineMath math={`\\mathbf{A} = \\begin{pmatrix}${attnRef.A[0].map(v=>v.toFixed(3)).join('&')}\\\\${attnRef.A[1].map(v=>v.toFixed(3)).join('&')}\\end{pmatrix}`} /></p>
                    <p style={{ fontSize: '11px', color: '#64748b' }}>
                      Softmax ligne 2 : [e¹/(e¹+e¹), e¹/(e¹+e¹)] = [0.5, 0.5] (scores égaux → distribution uniforme).
                    </p>
                  </div>
                )}
              </div>

              {/* Visual: Heatmap computed from real values */}
              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Heatmap d'attention 2×2</span>
                </div>
                <div className="mini-heatmap-grid">
                  {attnRef.A.map((row, i) =>
                    row.map((v, j) => (
                      <div
                        key={`${i}-${j}`}
                        className="heat-cell"
                        style={{ backgroundColor: `rgba(37, 99, 235, ${v})`, color: v > 0.5 ? '#fff' : '#1e3a8a' }}
                        title={`A[${i+1},${j+1}] = ${v.toFixed(3)}`}
                      >
                        {(v * 100).toFixed(0)}%
                      </div>
                    ))
                  )}
                </div>
                <p style={{ fontSize: '11px', color: '#64748b', padding: '8px 12px', textAlign: 'center' }}>
                  Ligne 1 : token 1 porte ~{(attnRef.A[0][0]*100).toFixed(0)}% sur token 1 et ~{(attnRef.A[0][1]*100).toFixed(0)}% sur token 2.
                  <br/>Ligne 2 : distribution uniforme (scores S[2,1]=S[2,2]=1).
                </p>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            ÉTAPE 6 — MLP & GELU (commune ViT & DeiT)
        ══════════════════════════════════════════════════════════════════ */}
        {currentStep === 6 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">ÉTAPE 06 / 07 — commune à ViT et DeiT</span>
              <h3>Réseau MLP et activation GELU</h3>
              <p>
                L'activation <InlineMath math="\text{GELU}(x) = x\,\Phi(x)" /> pondère l'entrée par la probabilité
                cumulée d'une loi normale standard. Contrairement à ReLU, GELU présente un léger creux pour x ≈ −0.75.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">6</span>
                  <span className="panel-header-title">Structure MLP & Valeurs GELU</span>
                </div>

                <div className="formula-callout-box">
                  <BlockMath math="\text{MLP}(\mathbf{z}) = \mathbf{W}_2\,\text{GELU}(\mathbf{W}_1\mathbf{z} + \mathbf{b}_1) + \mathbf{b}_2" />
                  <BlockMath math="\text{GELU}(x) \approx 0.5\,x\!\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\bigl(x + 0.044715\,x^3\bigr)\right)\right)" />
                </div>

                <div className="info-banner-blue" style={{ marginBottom: '10px', fontSize: '11.5px' }}>
                  <Info size={14} />
                  <span>
                    Structure : Linear(D→4D) → GELU → Linear(4D→D). En mode {isReal ? 'Réel' : 'Miniature'} : D = {isReal ? 768 : 4}, 4D = {isReal ? 3072 : 16}.
                  </span>
                </div>

                <div className="exercise-inputs-grid">
                  <div className="input-field-card">
                    <label>1. Facteur d'expansion de la couche cachée MLP :</label>
                    <input type="number" className="styled-lab-input"
                      value={step6Inputs.expansionFactor}
                      onChange={(e) => setStep6Inputs({ ...step6Inputs, expansionFactor: e.target.value })}
                      placeholder="Entrez un nombre" />
                  </div>
                  <div className="input-field-card">
                    <label>2. Dimension MLP (<InlineMath math={`4 \\times ${isReal ? 768 : 4}`} />) :</label>
                    <input type="number" className="styled-lab-input"
                      value={step6Inputs.mlpDim}
                      onChange={(e) => setStep6Inputs({ ...step6Inputs, mlpDim: e.target.value })}
                      placeholder="Entrez un nombre" />
                  </div>
                  <div className="sub-calc-step">
                    <span className="grp-label">3. Calcul de GELU (formule exacte) :</span>
                    <div className="result-cells-row">
                      <span>GELU(3.0) ≈ </span>
                      <input type="number" step="0.001" className="cell-input"
                        value={step6Inputs.geluZ1}
                        onChange={(e) => setStep6Inputs({ ...step6Inputs, geluZ1: e.target.value })}
                        placeholder="?" />
                      <span>GELU(−1.0) ≈ </span>
                      <input type="number" step="0.001" className="cell-input"
                        value={step6Inputs.geluZ2}
                        onChange={(e) => setStep6Inputs({ ...step6Inputs, geluZ2: e.target.value })}
                        placeholder="?" />
                    </div>
                  </div>
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep6}>
                    <CheckCircle2 size={16} /><span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-hint" onClick={() => showNextHint(6)}>
                    <HelpCircle size={15} /><span>Indice {hintLevels[6] ? `(${hintLevels[6]}/3)` : ''}</span>
                  </button>
                  <button type="button" className="btn-toggle-solution" onClick={() => setSolutionVisible({ ...solutionVisible, 6: !solutionVisible[6] })}>
                    <span>{solutionVisible[6] ? 'Masquer' : 'Voir la correction'}</span>
                  </button>
                </div>

                {step6Feedback && (
                  <div className={`feedback-alert-box ${step6Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step6Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step6Feedback.message}</span>
                  </div>
                )}

                {hintLevels[6] >= 1 && (
                  <div className="hint-collapsible-box">
                    <strong>Indice {hintLevels[6]} :</strong>{' '}
                    {hintLevels[6] === 1 && 'Le MLP de ViT-Base expandit les activations 4× en largeur (D→4D) avant de les comprimer (4D→D).'}
                    {hintLevels[6] === 2 && 'Pour GELU(3.0) : x très positif → Φ(x)≈1 → GELU(x)≈x. Pour GELU(−1) : x négatif → Φ(x)<0.5 → GELU<0.'}
                    {hintLevels[6] >= 3 && `GELU(3.0) = 0.5×3×(1+tanh(√(2/π)×(3+0.044715×27))) ≈ ${geluExactLocal(3.0).toFixed(3)}. GELU(−1.0) ≈ ${geluExactLocal(-1.0).toFixed(3)}.`}
                  </div>
                )}

                {solutionVisible[6] && (
                  <div className="solution-collapsible-box">
                    <h4>Correction :</h4>
                    <p>Facteur = 4. D_MLP = 4 × {isReal ? 768 : 4} = {isReal ? 3072 : 16}.</p>
                    <p>GELU(3.0) = {geluExactLocal(3.0).toFixed(3)} (≈ x pour x ≫ 0)</p>
                    <p>GELU(−1.0) = {geluExactLocal(-1.0).toFixed(3)} (creux léger, contrairement à ReLU qui donne 0)</p>
                  </div>
                )}
              </div>

              {/* Visual: Corrected GELU curve */}
              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Courbe GELU vs ReLU (axe correct)</span>
                </div>
                <GELUChart />
                <div style={{ padding: '10px 12px' }}>
                  <EncoderBlockDiagram />
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            ÉTAPE 7 — CLASSIFICATION FINALE
        ══════════════════════════════════════════════════════════════════ */}
        {currentStep === 7 && (
          <div className="lab-step-view">
            <div className="step-objective-banner">
              <span className="step-kicker">
                ÉTAPE 07 / 07 — {selectedModel === 'vit' ? 'ViT : tête [CLS] unique' : 'DeiT : deux têtes [CLS] + [DIST]'}
              </span>
              <h3>Classification Softmax finale {selectedModel === 'deit' ? '+ Fusion de Distillation DeiT' : '(ViT)'}</h3>
              <p>
                Avant la tête : LN final appliqué sur le token [CLS]{selectedModel === 'deit' ? ' et [DIST]' : ''}.
                Logits ViT/[CLS] : <InlineMath math="\mathbf{z}_{\text{cls}} = [2.0, 1.0, 0.0]" />.
              </p>
            </div>

            <div className="lab-two-columns">
              <div className="lab-panel-box">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">7</span>
                  <span className="panel-header-title">
                    {selectedModel === 'vit' ? 'Softmax ViT — tête [CLS]' : 'Softmax + Fusion DeiT — deux têtes'}
                  </span>
                </div>

                <div className="formula-callout-box">
                  {selectedModel === 'deit' ? (
                    <>
                      <BlockMath math="P_{\text{cls}}(y{=}i) = \frac{e^{z_i^{\text{cls}}}}{\sum_j e^{z_j^{\text{cls}}}}, \quad P_{\text{dist}}(y{=}i) = \frac{e^{z_i^{\text{dist}}}}{\sum_j e^{z_j^{\text{dist}}}}" />
                      <BlockMath math="P_{\text{final}} = \tfrac{1}{2}\bigl(P_{\text{cls}} + P_{\text{dist}}\bigr) \quad \text{(fusion papier DeiT)}" />
                    </>
                  ) : (
                    <BlockMath math="P_{\text{ViT}}(y{=}i) = \frac{e^{z_i^{\text{cls}}}}{\sum_j e^{z_j^{\text{cls}}}}, \qquad \hat{y} = \operatorname{argmax}_i P_{\text{ViT}}(y{=}i)" />
                  )}
                </div>

                {/* LN final note */}
                <div className="info-banner-blue" style={{ marginBottom: '10px', fontSize: '11.5px' }}>
                  <Info size={14} />
                  <span>
                    <strong>LN final :</strong> <InlineMath math="y = \text{LN}(z_L^0)" /> appliqué sur le vecteur du token [CLS] avant la tête linéaire D→1000.
                    {selectedModel === 'deit' ? ' De même pour [DIST].' : ''}
                    <br/>Convention : « classe 1 » = premier logit (indexation à partir de 1).
                  </span>
                </div>

                <div className="exercise-inputs-grid">
                  {/* Common: softmax of z_cls=[2,1,0] */}
                  <div className="sub-calc-step">
                    <span className="grp-label">
                      1. {selectedModel === 'deit' ? 'Tête [CLS] — exponentielles' : 'Exponentielles des logits'} :
                    </span>
                    <p className="desc-text" style={{ fontSize: '11px' }}>
                      z_cls = [2.0, 1.0, 0.0]. Calculez exp(z_cls[1]) et la somme totale.
                    </p>
                    <div className="result-cells-row">
                      <span>exp(2.0) = </span>
                      <input type="number" step="0.001" className="cell-input"
                        value={step7Inputs.exp1}
                        onChange={(e) => setStep7Inputs({ ...step7Inputs, exp1: e.target.value })}
                        placeholder="?" />
                      <span>Σexp = </span>
                      <input type="number" step="0.01" className="cell-input"
                        value={step7Inputs.sumExp}
                        onChange={(e) => setStep7Inputs({ ...step7Inputs, sumExp: e.target.value })}
                        placeholder="?" />
                    </div>
                  </div>

                  <div className="input-field-card">
                    <label>2. Probabilité classe 1 — <InlineMath math="P_{\text{cls}}(y=1)" /> :</label>
                    <input type="number" step="0.001" className="styled-lab-input"
                      value={step7Inputs.prob1}
                      onChange={(e) => setStep7Inputs({ ...step7Inputs, prob1: e.target.value })}
                      placeholder="?" />
                  </div>

                  {selectedModel === 'deit' ? (
                    <div className="sub-calc-step deit-highlight-box">
                      <span className="grp-label">
                        3. Fusion papier DeiT — <InlineMath math="P_{\text{dist}}" /> donnée :{' '}
                        <InlineMath math={`[${step7Ref.softmaxDist.probs.map(v=>v.toFixed(3)).join(',\\; ')}]`} />
                      </span>
                      <p className="desc-text">
                        Calculez <InlineMath math="P_{\text{final}} = \tfrac{1}{2}(P_{\text{cls}} + P_{\text{dist}})" /> pour les 3 classes :
                      </p>
                      <div className="result-cells-row">
                        {['P_f₁ = ', 'P_f₂ = ', 'P_f₃ = '].map((label, i) => {
                          const keys = ['pFinalPaper1', 'pFinalPaper2', 'pFinalPaper3'] as const;
                          return (
                            <React.Fragment key={i}>
                              <span>{label}</span>
                              <input type="number" step="0.001" className="cell-input"
                                value={step7Inputs[keys[i]]}
                                onChange={(e) => setStep7Inputs({ ...step7Inputs, [keys[i]]: e.target.value })}
                                placeholder="?" />
                            </React.Fragment>
                          );
                        })}
                      </div>
                    </div>
                  ) : (
                    <div className="vit-head-notice-box">
                      <span className="notice-pill">Architecture ViT — Tête unique [CLS]</span>
                      <p className="desc-text">
                        Dans Google ViT, la décision finale repose à <strong>100% sur le token [CLS]</strong>.
                        Il n'y a pas de token [DIST] ni de seconde tête. Pour explorer la distillation, basculez sur <strong>DeiT</strong>.
                      </p>
                    </div>
                  )}
                </div>

                <div className="exercise-action-buttons">
                  <button type="button" className="btn-verify-answer" onClick={verifyStep7}>
                    <CheckCircle2 size={16} /><span>Vérifier ma réponse</span>
                  </button>
                  <button type="button" className="btn-toggle-hint" onClick={() => showNextHint(7)}>
                    <HelpCircle size={15} /><span>Indice {hintLevels[7] ? `(${hintLevels[7]}/3)` : ''}</span>
                  </button>
                  <button type="button" className="btn-toggle-solution" onClick={() => setSolutionVisible({ ...solutionVisible, 7: !solutionVisible[7] })}>
                    <span>{solutionVisible[7] ? 'Masquer' : 'Voir la solution'}</span>
                  </button>
                </div>

                {step7Feedback && (
                  <div className={`feedback-alert-box ${step7Feedback.isCorrect ? 'success' : 'error'}`}>
                    {step7Feedback.isCorrect ? <CheckCircle2 size={18} /> : <XCircle size={18} />}
                    <span>{step7Feedback.message}</span>
                  </div>
                )}

                {hintLevels[7] >= 1 && (
                  <div className="hint-collapsible-box">
                    <strong>Indice {hintLevels[7]} :</strong>{' '}
                    {hintLevels[7] === 1 && 'exp(2.0) = e² ≈ 7.389, exp(1.0) ≈ 2.718, exp(0) = 1.'}
                    {hintLevels[7] === 2 && `Σexp = ${step7Ref.softmaxCls.sumExp.toFixed(3)}, P₁ = 7.389 / ${step7Ref.softmaxCls.sumExp.toFixed(3)} ≈ ${step7Ref.softmaxCls.probs[0].toFixed(3)}.`}
                    {hintLevels[7] >= 3 && selectedModel === 'deit'
                      ? `P_cls=[${step7Ref.softmaxCls.probs.map(v=>v.toFixed(3)).join(', ')}], P_dist=[${step7Ref.softmaxDist.probs.map(v=>v.toFixed(3)).join(', ')}]. Fusion: [${step7Ref.pFinalPaper.map(v=>v.toFixed(3)).join(', ')}].`
                      : `P₁≈${step7Ref.softmaxCls.probs[0].toFixed(3)} (classe 1 gagne, 1-indexé).`}
                  </div>
                )}

                {solutionVisible[7] && (
                  <div className="solution-collapsible-box">
                    <h4>Correction complète :</h4>
                    <p><InlineMath math={`e^2 = ${step7Ref.softmaxCls.exps[0].toFixed(3)},\\; e^1 = ${step7Ref.softmaxCls.exps[1].toFixed(3)},\\; e^0 = 1.000`} /></p>
                    <p><InlineMath math={`\\Sigma = ${step7Ref.softmaxCls.sumExp.toFixed(3)},\\; P_1 = ${step7Ref.softmaxCls.probs[0].toFixed(3)}`} /></p>
                    {selectedModel === 'deit' && (
                      <>
                        <p><strong>z_dist=[1,3,0] :</strong></p>
                        <p><InlineMath math={`P_{\\text{dist}} = [${step7Ref.softmaxDist.probs.map(v=>v.toFixed(3)).join(',\\; ')}]`} /></p>
                        <p><InlineMath math={`P_{\\text{final}} = [${step7Ref.pFinalPaper.map(v=>v.toFixed(3)).join(',\\; ')}]`} /> → classe {step7Ref.winnerPaper}</p>
                        <p style={{ fontSize: '11px', color: '#6d28d9', marginTop: '8px' }}>
                          <strong>Variante HuggingFace</strong> (moyenne des logits) : softmax([1.5, 2, 0]) ≈ [{step7Ref.pFinalHF.map(v=>v.toFixed(3)).join(', ')}] → classe {step7Ref.winnerHF} (même gagnant ici).
                        </p>
                      </>
                    )}
                  </div>
                )}

                {/* DeiT: Distillation Loss section */}
                {selectedModel === 'deit' && (
                  <div style={{
                    marginTop: '14px',
                    background: '#faf5ff',
                    border: '1px solid #c4b5fd',
                    borderRadius: '8px',
                    padding: '12px 14px',
                  }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                      <BookOpen size={15} style={{ color: '#6d28d9' }} />
                      <strong style={{ fontSize: '12px', color: '#4c1d95' }}>Perte de distillation (lecture seule)</strong>
                    </div>
                    <div style={{ fontSize: '12px', color: '#5b21b6', lineHeight: '1.6' }}>
                      <BlockMath math="\mathcal{L} = \tfrac{1}{2}\,\text{CE}\!\left(\psi(Z_{\text{cls}}),\, y\right) + \tfrac{1}{2}\,\text{CE}\!\left(\psi(Z_{\text{dist}}),\, y_t\right)" />
                      <p>y = vrai label (supervision ImageNet). <strong>y_t = argmax</strong> du logit de l'enseignant CNN (RegNetY-16GF dans le papier).</p>
                      <p><strong>[CLS]</strong> apprend des vrais labels humains. <strong>[DIST]</strong> imite les décisions de l'enseignant.</p>
                      <p style={{ fontSize: '11px', color: '#7c3aed', marginTop: '4px' }}>
                        Version douce (température τ) : remplacer CE par KL-divergence entre softmax(Z_dist/τ) et softmax(z_teacher/τ).
                      </p>
                    </div>
                  </div>
                )}
              </div>

              {/* Visual: dual-branch recap */}
              <div className="lab-panel-box visual-panel">
                <div className="panel-navy-header">
                  <span className="panel-num-circle">★</span>
                  <span className="panel-header-title">Pipeline End-to-End</span>
                </div>
                <div style={{ padding: '12px 10px' }}>
                  <PipelineRecap model={selectedModel as 'vit' | 'deit'} />

                  {/* DeiT variante HF comparison */}
                  {selectedModel === 'deit' && (
                    <div style={{
                      marginTop: '14px',
                      display: 'grid',
                      gridTemplateColumns: '1fr 1fr',
                      gap: '10px',
                    }}>
                      <div style={{ background: '#eff6ff', border: '1px solid #93c5fd', borderRadius: '8px', padding: '10px' }}>
                        <div style={{ fontSize: '11px', fontWeight: '700', color: '#1e40af', marginBottom: '4px' }}>Fusion papier DeiT</div>
                        <div style={{ fontSize: '10.5px', color: '#1e40af' }}>
                          P_final = ½(P_cls + P_dist)
                          <br/>[{step7Ref.pFinalPaper.map(v=>`${(v*100).toFixed(1)}%`).join(', ')}]
                          <br/><strong>Classe {step7Ref.winnerPaper}</strong>
                        </div>
                      </div>
                      <div style={{ background: '#f5f3ff', border: '1px solid #c4b5fd', borderRadius: '8px', padding: '10px' }}>
                        <div style={{ fontSize: '11px', fontWeight: '700', color: '#6d28d9', marginBottom: '4px' }}>Variante HuggingFace</div>
                        <div style={{ fontSize: '10.5px', color: '#6d28d9' }}>
                          softmax((z_cls + z_dist)/2)
                          <br/>[{step7Ref.pFinalHF.map(v=>`${(v*100).toFixed(1)}%`).join(', ')}]
                          <br/><strong>Classe {step7Ref.winnerHF}</strong>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

      </div>

      {/* ── NAV FOOTER ── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '12px' }}>
        <button
          type="button"
          className="btn-studio-secondary"
          onClick={() => setCurrentStep(Math.max(1, currentStep - 1))}
          disabled={currentStep === 1}
          style={{ opacity: currentStep === 1 ? 0.4 : 1 }}
        >
          <ArrowLeft size={16} /><span>Étape précédente</span>
        </button>
        <button
          type="button"
          className="btn-verify-answer"
          onClick={() => setCurrentStep(Math.min(7, currentStep + 1))}
          disabled={currentStep === 7}
          style={{ opacity: currentStep === 7 ? 0.4 : 1 }}
        >
          <span>Étape suivante</span><ArrowRight size={16} />
        </button>
      </div>
    </div>
  );
}
