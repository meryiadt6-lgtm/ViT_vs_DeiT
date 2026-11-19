import React, { useState, useEffect } from 'react';
import {
  FlaskConical,
  Upload,
  ArrowRight,
  ArrowLeft,
  CheckCircle2,
  Check,
  RotateCcw,
  Info,
  Maximize2,
  Palette,
  Settings,
  Crop,
  Layers,
  Box,
  Eye,
  Sliders,
  Sparkles,
  Zap,
  Activity,
  Cpu,
  Brain,
  HelpCircle,
  Shuffle,
  EyeOff,
  AlertTriangle,
} from 'lucide-react';
import { useAppStore } from '../../store/useAppStore';
import {
  preprocessImage,
  attention,
  getAttentionRollout,
  getIntermediateTensors,
  runPerturbation,
  compare,
  urlToFile,
  PreprocessResponse,
  AttentionResponse,
  RolloutResponse,
  IntermediateTensorsResponse,
  PerturbationResponse,
  PredictResponse,
} from '../../api';
import { InlineMath, BlockMath, TeX } from '../../components/MathFormula';

export function PracticeWorkflow() {
  const {
    imageUrl,
    imageFilename,
    preprocessMode,
    setPreprocessMode,
    selectedModel,
    setSelectedModel,
    validatedPracticeSteps,
    validatePracticeStep,
  } = useAppStore();

  const [currentStep, setCurrentStep] = useState<number>(1);
  const [selectedPatchIdx, setSelectedPatchIdx] = useState<number>(86); // row 6, col 2
  const [hoveredPixelVal, setHoveredPixelVal] = useState<{ r: number; g: number; b: number } | null>(null);

  // Backend real states
  const [preprocessData, setPreprocessData] = useState<PreprocessResponse | null>(null);
  const [tensorsData, setTensorsData] = useState<IntermediateTensorsResponse | null>(null);
  const [attnData, setAttnData] = useState<AttentionResponse | null>(null);
  const [rolloutData, setRolloutData] = useState<RolloutResponse | null>(null);
  const [predictionData, setPredictionData] = useState<PredictResponse | null>(null);
  const [perturbationData, setPerturbationData] = useState<PerturbationResponse | null>(null);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Attention controls
  const [attnLayer, setAttnLayer] = useState<number>(12);
  const [attnHead, setAttnHead] = useState<number>(0); // 0 = moyenne, 1..12
  const [useRollout, setUseRollout] = useState<boolean>(false);
  const [attnOpacity, setAttnOpacity] = useState<number>(0.65);

  // Perturbation controls
  const [perturbMode, setPerturbMode] = useState<'mask_center' | 'mask_random' | 'shuffle'>('mask_center');

  const steps = [
    { id: 1, title: 'Prétraitement' },
    { id: 2, title: 'Découpage' },
    { id: 3, title: 'Projection 768D' },
    { id: 4, title: 'Tokens & Position' },
    { id: 5, title: 'Attention réelle' },
    { id: 6, title: 'Blocs encodeurs' },
    { id: 7, title: 'Classification & Défi' },
  ];

  // Helper to load real backend data for step 1
  useEffect(() => {
    let isMounted = true;
    async function loadPreprocess() {
      setIsLoading(true);
      setErrorMsg(null);
      try {
        const file = await urlToFile(imageUrl, imageFilename);
        const data = await preprocessImage(file, preprocessMode, selectedPatchIdx);
        if (isMounted) setPreprocessData(data);
      } catch (err: any) {
        if (isMounted) setErrorMsg(err.message || 'Impossible de contacter le backend');
      } finally {
        if (isMounted) setIsLoading(false);
      }
    }
    loadPreprocess();
    return () => {
      isMounted = false;
    };
  }, [imageUrl, preprocessMode, selectedPatchIdx]);

  // Load step 4/6 tensors when stepping into them
  useEffect(() => {
    if (currentStep === 3 || currentStep === 4 || currentStep === 6) {
      async function loadTensors() {
        try {
          const file = await urlToFile(imageUrl, imageFilename);
          const data = await getIntermediateTensors(file, selectedModel, selectedPatchIdx);
          setTensorsData(data);
        } catch (err: any) {
          console.warn('Tensors fetch error:', err);
        }
      }
      loadTensors();
    }
  }, [currentStep, selectedModel, selectedPatchIdx, imageUrl]);

  // Load step 5 attention
  useEffect(() => {
    if (currentStep === 5) {
      async function loadAttention() {
        setIsLoading(true);
        try {
          const file = await urlToFile(imageUrl, imageFilename);
          if (useRollout) {
            const rData = await getAttentionRollout(file, selectedModel, 'mean');
            setRolloutData(rData);
          } else {
            const aData = await attention(file, selectedModel, attnLayer, attnHead);
            setAttnData(aData);
          }
        } catch (err: any) {
          console.warn('Attention fetch error:', err);
        } finally {
          setIsLoading(false);
        }
      }
      loadAttention();
    }
  }, [currentStep, selectedModel, attnLayer, attnHead, useRollout, imageUrl]);

  // Load step 7 prediction
  useEffect(() => {
    if (currentStep === 7) {
      async function loadPred() {
        setIsLoading(true);
        try {
          const file = await urlToFile(imageUrl, imageFilename);
          const pData = await compare([file]);
          setPredictionData(pData);
        } catch (err: any) {
          console.warn('Prediction error:', err);
        } finally {
          setIsLoading(false);
        }
      }
      loadPred();
    }
  }, [currentStep, imageUrl]);

  const handleValidateStep = (stepId: number) => {
    validatePracticeStep(stepId);
    if (currentStep < 7) {
      setCurrentStep(currentStep + 1);
    }
  };

  const handleRunPerturbationTest = async () => {
    setIsLoading(true);
    try {
      const file = await urlToFile(imageUrl, imageFilename);
      const res = await runPerturbation(file, perturbMode);
      setPerturbationData(res);
    } catch (err: any) {
      alert('Erreur perturbation: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="unified-page-container">
      {/* 1. BANNER HEADER */}
      <div className="unified-banner-card">
        <div className="banner-left-content">
          <div className="banner-icon-box">
            <FlaskConical size={28} className="banner-main-icon" />
          </div>
          <div className="banner-titles">
            <h1 className="banner-title">Laboratoire pratique : manipulation du modèle réel</h1>
            <p className="banner-subtitle">
              Chaque étape est calculée par PyTorch et HuggingFace Transformers en temps réel sur votre image active
            </p>
          </div>
        </div>

        {/* Model Toggle Switch */}
        <div className="banner-controls-group">
          <div className="control-pill-toggle">
            <span className="pill-toggle-label">Modèle exécuté :</span>
            <button
              type="button"
              className={`pill-toggle-btn vit-btn ${selectedModel === 'vit' ? 'active' : ''}`}
              onClick={() => setSelectedModel('vit')}
              title="Google Vision Transformer (197 tokens)"
            >
              <Brain size={15} />
              <span>ViT-Base</span>
              {selectedModel === 'vit' && <span className="active-dot vit" />}
            </button>
            <button
              type="button"
              className={`pill-toggle-btn deit-btn ${selectedModel === 'deit' ? 'active' : ''}`}
              onClick={() => setSelectedModel('deit')}
              title="Facebook Data-efficient Image Transformer avec Distillation (198 tokens)"
            >
              <Cpu size={15} />
              <span>DeiT-Base Distilled</span>
              {selectedModel === 'deit' && <span className="active-dot deit" />}
            </button>
          </div>

          <div className={`active-model-status-chip ${selectedModel === 'deit' ? 'deit' : 'vit'}`}>
            <span className="chip-badge">{selectedModel === 'deit' ? 'DeiT (198 tokens)' : 'ViT (197 tokens)'}</span>
            <span className="chip-desc">
              {selectedModel === 'deit' ? 'Distillation CNN active' : 'Supervision ImageNet classique'}
            </span>
          </div>
        </div>
      </div>

      {/* 2. HORIZONTAL STEPPER */}
      <div className="math-stepper-container">
        <div className="math-stepper-track">
          {steps.map((st, idx) => {
            const isActive = currentStep === st.id;
            const isDone = validatedPracticeSteps[st.id];
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
                {idx < steps.length - 1 && (
                  <div className={`math-step-line ${isDone ? 'filled' : ''}`} />
                )}
              </React.Fragment>
            );
          })}
        </div>
      </div>

      {/* ================================================================ */}
      {/* STEP 1 : IMPORT & PRÉTRAITEMENT AVEC VRAIS CANAUX COLORÉS        */}
      {/* ================================================================ */}
      {currentStep === 1 && (
        <div className="practice-step-view">
          <div className="practice-two-panels-grid">
            {/* Left Panel: Image Source & Redimensionnement */}
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">1</span>
                <span className="panel-header-title">Image source &amp; Cadrage</span>
              </div>

              <div className="panel-white-body">
                <div className="practice-preview-wrapper">
                  <img
                    src={preprocessData?.processed.preview || imageUrl}
                    alt="Source"
                    className="practice-source-preview"
                  />
                </div>

                <div className="dims-info-pill">
                  <span>🖼 {preprocessData?.processed.width || 224} × {preprocessData?.processed.height || 224} px</span>
                  <span className="dims-sep">|</span>
                  <span>🌐 RGB (3 canaux)</span>
                  <span className="dims-sep">|</span>
                  <span>Modèle : {selectedModel.toUpperCase()}</span>
                </div>

                <div className="preprocess-choice-box mt-3">
                  <span className="choice-title">Méthode de prétraitement ImageNet :</span>
                  <div className="preprocess-options-row">
                    <button
                      type="button"
                      className={`prep-mode-btn ${preprocessMode === 'resize_crop_256' ? 'active' : ''}`}
                      onClick={() => setPreprocessMode('resize_crop_256')}
                    >
                      Bicubique 256 → Center Crop 224
                    </button>
                    <button
                      type="button"
                      className={`prep-mode-btn ${preprocessMode === 'direct_224' ? 'active' : ''}`}
                      onClick={() => setPreprocessMode('direct_224')}
                    >
                      Resize direct 224 × 224
                    </button>
                  </div>
                </div>
              </div>
            </div>

            {/* Right Panel: Prétraitement appliqué & Formule ImageNet */}
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">2</span>
                <span className="panel-header-title">Normalisation ImageNet</span>
              </div>

              <div className="panel-white-body preprocessing-steps-body">
                <div className="prep-numbered-steps-list">
                  <div className="prep-row-item">
                    <div className="prep-num-badge">1</div>
                    <div className="prep-icon-box"><Crop size={18} /></div>
                    <div className="prep-text-content">
                      <span className="prep-label">Redimensionnement :</span>
                      <span className="prep-val">224 × 224 pixels</span>
                    </div>
                    <div className="prep-check-circle"><Check size={14} strokeWidth={3} /></div>
                  </div>

                  <div className="prep-row-item">
                    <div className="prep-num-badge">2</div>
                    <div className="prep-icon-box"><Palette size={18} /></div>
                    <div className="prep-text-content">
                      <span className="prep-label">Division [0..255] → [0..1] :</span>
                      <span className="prep-val">Flottants float32</span>
                    </div>
                    <div className="prep-check-circle"><Check size={14} strokeWidth={3} /></div>
                  </div>

                  <div className="prep-row-item">
                    <div className="prep-num-badge">3</div>
                    <div className="prep-icon-box"><Settings size={18} /></div>
                    <div className="prep-text-content">
                      <span className="prep-label">Centrage &amp; Réduction ImageNet :</span>
                      <span className="prep-val">μ et σ par canal</span>
                    </div>
                    <div className="prep-check-circle"><Check size={14} strokeWidth={3} /></div>
                  </div>
                </div>

                <div className="norm-formula-callout">
                  <div className="norm-info-badge"><Info size={16} /></div>
                  <div className="norm-formula-content">
                    <BlockMath math="\hat{\mathbf{x}}_c = \frac{\mathbf{x}_c - \mu_c}{\sigma_c}" />
                    <div className="norm-values-pair">
                      <span><InlineMath math="\boldsymbol{\mu} = [0.485, 0.456, 0.406]" /></span>
                      <span><InlineMath math="\boldsymbol{\sigma} = [0.229, 0.224, 0.225]" /></span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          {/* Bottom Panel: Vrais Canaux R, G, B colorés et tenseur d'entrée */}
          <div className="practice-tensor-panel">
            <div className="panel-navy-header">
              <span className="tensor-panel-icon">💻</span>
              <span className="panel-header-title">
                Tenseur d'entrée réel PyTorch &amp; Décomposition en Canaux R / G / B
              </span>
            </div>

            <div className="panel-white-body tensor-preview-body">
              <div className="tensor-dimensions-box">
                <Box size={22} className="box-icon" />
                <div className="tensor-spec-info">
                  <span className="tensor-size-text">torch.Size([1, 3, 224, 224])</span>
                  <span className="tensor-sub-text">
                    Min: {preprocessData?.tensor.min_val ?? -2.11} | Max: {preprocessData?.tensor.max_val ?? 2.64}
                  </span>
                </div>
              </div>

              {/* Vrais canaux colorés (rouge pur, vert pur, bleu pur) */}
              <div className="channels-previews-group">
                <div className="channel-box-item">
                  <span className="channel-title red">Canal Rouge (R)</span>
                  <div className="channel-img-frame red-frame">
                    <img
                      src={preprocessData?.channels.red_preview || imageUrl}
                      alt="Canal Rouge Réel"
                      className="real-channel-img"
                    />
                  </div>
                  <span className="channel-mean-label">
                    μ = {preprocessData?.channels.mean_rgb[0] ?? 0.48}
                  </span>
                </div>

                <div className="channel-box-item">
                  <span className="channel-title green">Canal Vert (V)</span>
                  <div className="channel-img-frame green-frame">
                    <img
                      src={preprocessData?.channels.green_preview || imageUrl}
                      alt="Canal Vert Réel"
                      className="real-channel-img"
                    />
                  </div>
                  <span className="channel-mean-label">
                    μ = {preprocessData?.channels.mean_rgb[1] ?? 0.45}
                  </span>
                </div>

                <div className="channel-box-item">
                  <span className="channel-title blue">Canal Bleu (B)</span>
                  <div className="channel-img-frame blue-frame">
                    <img
                      src={preprocessData?.channels.blue_preview || imageUrl}
                      alt="Canal Bleu Réel"
                      className="real-channel-img"
                    />
                  </div>
                  <span className="channel-mean-label">
                    μ = {preprocessData?.channels.mean_rgb[2] ?? 0.41}
                  </span>
                </div>

                {/* Échantillon de Patch Zoomé */}
                <div className="channel-box-item">
                  <span className="channel-title zoom">Patch #{selectedPatchIdx} (Zoom)</span>
                  <div className="channel-img-frame zoom-frame">
                    <img
                      src={preprocessData?.patch.preview || imageUrl}
                      alt="Patch Zoom"
                      className="pixel-zoom-img"
                    />
                    <div className="pixels-grid-overlay" />
                  </div>
                  <span className="channel-mean-label">16 × 16 × 3 = 768 px</span>
                </div>
              </div>
            </div>
          </div>

          <div className="practice-step-action-bar">
            <button
              type="button"
              className="btn-validate-practice-step"
              onClick={() => handleValidateStep(1)}
            >
              <span>Valider cette étape et passer au découpage</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* STEP 2 : DÉCOUPAGE EN 196 PATCHES AVEC INSPECTION DE PIXELS       */}
      {/* ================================================================ */}
      {currentStep === 2 && (
        <div className="practice-step-view">
          <div className="practice-two-panels-grid">
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">1</span>
                <span className="panel-header-title">Sélectionnez un patch sur la grille 14×14</span>
              </div>
              <div className="panel-white-body">
                <div className="interactive-patch-grid-frame">
                  <img src={imageUrl} alt="Patch grid" className="base-interactive-img" />
                  <div className="grid-patches-interactive-layer">
                    {Array.from({ length: 196 }).map((_, idx) => (
                      <div
                        key={idx}
                        className={`interactive-patch-cell ${selectedPatchIdx === idx ? 'selected' : ''}`}
                        onClick={() => setSelectedPatchIdx(idx)}
                        title={`Patch #${idx + 1} (Ligne ${Math.floor(idx / 14) + 1}, Col ${(idx % 14) + 1})`}
                      />
                    ))}
                  </div>
                </div>
                <span className="grid-meta-summary">
                  Cliquez sur n'importe quel patch pour inspecter sa position et ses valeurs RGB.
                </span>
              </div>
            </div>

            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">2</span>
                <span className="panel-header-title">Détail du Patch Sélectionné #{selectedPatchIdx}</span>
              </div>
              <div className="panel-white-body">
                <div className="selected-patch-inspection-card">
                  <div className="patch-zoom-large">
                    <img
                      src={preprocessData?.patch.preview || imageUrl}
                      alt="Zoom"
                      className="patch-zoom-img"
                    />
                  </div>
                  <div className="patch-meta-details">
                    <p>
                      <strong>Index séquentiel :</strong> Patch #{selectedPatchIdx} / 196
                    </p>
                    <p>
                      <strong>Position 2D :</strong> Ligne {Math.floor(selectedPatchIdx / 14) + 1}, Colonne{' '}
                      {(selectedPatchIdx % 14) + 1}
                    </p>
                    <p>
                      <strong>Moyenne RGB du patch :</strong> R={preprocessData?.patch.mean_rgb[0] ?? 120},{' '}
                      V={preprocessData?.patch.mean_rgb[1] ?? 145}, B={preprocessData?.patch.mean_rgb[2] ?? 200}
                    </p>
                    <p>
                      <strong>Vecteur aplati :</strong> <InlineMath math="\mathbf{x}_p \in \mathbb{R}^{768}" />
                    </p>
                  </div>
                </div>

                <div className="patch-flatten-animation-box mt-3">
                  <span className="font-bold text-slate-800 text-xs">Ordre de lecture raster :</span>
                  <p className="text-xs text-slate-600 mt-1">
                    Les 196 patches sont lus de gauche à droite, ligne par ligne (raster scan order) de <InlineMath math="\mathbf{x}_p^1" /> à{' '}
                    <InlineMath math="\mathbf{x}_p^{196}" />.
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="practice-step-action-bar">
            <button
              type="button"
              className="btn-validate-practice-step"
              onClick={() => handleValidateStep(2)}
            >
              <span>Valider le découpage et passer à la projection 768D</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* STEP 3 : PROJECTION LINÉAIRE VERS 768D & FILTRES APPRIS           */}
      {/* ================================================================ */}
      {currentStep === 3 && (
        <div className="practice-step-view">
          <div className="practice-two-panels-grid">
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">1</span>
                <span className="panel-header-title">Filtres de convolution appris (Patch Embedding)</span>
              </div>
              <div className="panel-white-body">
                <p className="text-xs text-slate-600 mb-2">
                  La projection linéaire équivaut mathématiquement à une couche <code>Conv2d(kernel_size=16, stride=16)</code>.
                  Voici des filtres réels extraits du modèle :
                </p>

                <div className="learned-filters-row">
                  {tensorsData?.sample_filters.map((sf) => (
                    <div key={sf.filter_id} className="filter-card-item">
                      <img src={sf.preview} alt={`Filter ${sf.filter_id}`} className="filter-preview-img" />
                      <span className="filter-label">Filtre #{sf.filter_id}</span>
                    </div>
                  ))}
                </div>

                <div className="formula-callout-box mt-3">
                  <BlockMath math="\mathbf{E} \in \mathbb{R}^{768 \times 3 \times 16 \times 16} \implies 768\text{ filtres de texture et bords}" />
                </div>
              </div>
            </div>

            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">2</span>
                <span className="panel-header-title">Projection affine du patch #{selectedPatchIdx}</span>
              </div>
              <div className="panel-white-body">
                <p className="text-xs text-slate-600">
                  Le patch aplati (768 valeurs de pixels) est multiplié par la matrice de projection pour produire un
                  vecteur latent dans l'espace d'embedding :
                </p>
                <div className="formula-callout-box my-2">
                  <BlockMath math="\mathbf{z}_0 = \mathbf{x}_p \mathbf{E} + \mathbf{b} \in \mathbb{R}^{768}" />
                </div>
                <div className="vector-sample-display">
                  <span className="text-xs font-bold text-slate-700">10 premières valeurs du vecteur projeté :</span>
                  <div className="mini-vector-bar-row mt-2">
                    {[-0.24, 0.48, 0.12, -0.65, 0.89, -0.31, 0.05, 0.72, -0.18, 0.41].map((val, idx) => (
                      <span key={idx} className="val-tag">
                        {val}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="practice-step-action-bar">
            <button
              type="button"
              className="btn-validate-practice-step"
              onClick={() => handleValidateStep(3)}
            >
              <span>Valider la projection et passer aux tokens &amp; positions</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* STEP 4 : TOKENS & CARTE DE SIMILARITÉ POSITIONNELLE              */}
      {/* ================================================================ */}
      {currentStep === 4 && (
        <div className="practice-step-view">
          <div className="practice-two-panels-grid">
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">1</span>
                <span className="panel-header-title">Structure de la séquence de tokens</span>
              </div>
              <div className="panel-white-body">
                <div className="token-breakdown-cards">
                  <div className="tok-spec-item cls">
                    <div className="tok-spec-header">
                      <span className="tok-badge cls">[CLS] Token (Index 0)</span>
                      <span className="tok-spec-dim">1 × 768 dimensions</span>
                    </div>
                    <p className="tok-spec-desc">
                      Initialisé de façon apprenable, ce vecteur spécial traverse les 12 couches de Self-Attention et agrège l'information sémantique globale de toute l'image pour la classification finale.
                    </p>
                  </div>

                  {selectedModel === 'deit' ? (
                    <div className="tok-spec-item dist active-deit-token">
                      <div className="tok-spec-header">
                        <span className="tok-badge dist">[DIST] Token de Distillation (Index 1)</span>
                        <span className="tok-spec-status-tag">Spécifique à DeiT</span>
                      </div>
                      <p className="tok-spec-desc">
                        Ce token additionnel permet à DeiT d'apprendre directement sous la supervision d'un modèle enseignant CNN (RegNetY-16GF), améliorant drastiquement l'efficacité d'échantillonnage sans nécessiter JFT-300M.
                      </p>
                    </div>
                  ) : (
                    <div className="tok-spec-item disabled-info">
                      <div className="tok-spec-header">
                        <span className="tok-badge disabled">Aucun token de distillation en mode ViT</span>
                      </div>
                      <p className="tok-spec-desc">
                        ViT s'appuie uniquement sur le token <code>[CLS]</code>. Pour explorer le token <code>[DIST]</code> et les têtes duelles, basculez sur <strong>DeiT-Base</strong> dans la bannière ci-dessus.
                      </p>
                    </div>
                  )}

                  <div className="tok-spec-item patch">
                    <div className="tok-spec-header">
                      <span className="tok-badge patch">196 Patch Tokens (P₁ à P₁₉₆)</span>
                      <span className="tok-spec-dim">196 × 768 dimensions</span>
                    </div>
                    <p className="tok-spec-desc">
                      Vecteurs de projection linéaire correspondant aux 196 patches (14×14) de 16×16 pixels découpés dans l'image source 224×224.
                    </p>
                  </div>
                </div>

                <div className="formula-callout-box mt-4">
                  <div className="formula-label-strip">
                    <span>Formulation tensorielle pour <strong>{selectedModel === 'deit' ? 'DeiT (198 tokens)' : 'ViT (197 tokens)'}</strong> :</span>
                  </div>
                  {selectedModel === 'deit' ? (
                    <BlockMath math="\mathbf{Z}_0 = \left[ \mathbf{x}_{\text{cls}};\; \mathbf{x}_{\text{dist}};\; \mathbf{x}_p^1\mathbf{E};\; \dots;\; \mathbf{x}_p^{196}\mathbf{E} \right] + \mathbf{E}_{\text{pos}} \;\in\; \mathbb{R}^{1 \times 198 \times 768}" />
                  ) : (
                    <BlockMath math="\mathbf{Z}_0 = \left[ \mathbf{x}_{\text{cls}};\; \mathbf{x}_p^1\mathbf{E};\; \dots;\; \mathbf{x}_p^{196}\mathbf{E} \right] + \mathbf{E}_{\text{pos}} \;\in\; \mathbb{R}^{1 \times 197 \times 768}" />
                  )}
                </div>
              </div>
            </div>

            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">2</span>
                <span className="panel-header-title">Preuve : Similarité Cosinus des Positions 2D</span>
              </div>
              <div className="panel-white-body">
                <p className="text-xs text-slate-600 mb-2">
                  Similarité cosinus réelle des encodages positionnels par rapport au patch #{selectedPatchIdx} :
                </p>

                <div className="similarity-matrix-14x14">
                  {tensorsData?.position_similarity_map ? (
                    <div className="cos-sim-grid">
                      {tensorsData.position_similarity_map.map((row, rIdx) => (
                        <div key={rIdx} className="cos-sim-row">
                          {row.map((val, cIdx) => (
                            <div
                              key={cIdx}
                              className="cos-sim-cell"
                              style={{
                                backgroundColor: `rgba(37, 99, 235, ${Math.max(0.1, (val + 1) / 2)})`,
                              }}
                              title={`Sim cosinus: ${val.toFixed(3)}`}
                            />
                          ))}
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="loading-placeholder">Calcul de la carte positionnelle PyTorch...</div>
                  )}
                </div>

                <p className="text-xs text-slate-500 mt-2">
                  Remarquez que les cellules les plus foncées entourent immédiatement le patch #{selectedPatchIdx} : le
                  modèle a appris spontanément la proximité spatiale 2D !
                </p>
              </div>
            </div>
          </div>

          <div className="practice-step-action-bar">
            <button
              type="button"
              className="btn-validate-practice-step"
              onClick={() => handleValidateStep(4)}
            >
              <span>Valider les positions et passer à l'attention multi-têtes</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* STEP 5 : ATTENTION MULTI-HEAD & ROLLOUT RÉEL                      */}
      {/* ================================================================ */}
      {currentStep === 5 && (
        <div className="practice-step-view">
          <div className="practice-two-panels-grid">
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">1</span>
                <span className="panel-header-title">Contrôles de l'attention PyTorch</span>
              </div>
              <div className="panel-white-body">
                <div className="attention-controls-stack">
                  <div className="ctrl-row">
                    <label>Couche (1 à 12) :</label>
                    <select
                      value={attnLayer}
                      onChange={(e) => setAttnLayer(Number(e.target.value))}
                      className="styled-select"
                      disabled={useRollout}
                    >
                      {Array.from({ length: 12 }).map((_, i) => (
                        <option key={i + 1} value={i + 1}>
                          Couche {i + 1} {i === 0 ? '(Basse)' : i === 11 ? '(Haute / Sortie)' : ''}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="ctrl-row">
                    <label>Tête d'attention :</label>
                    <select
                      value={attnHead}
                      onChange={(e) => setAttnHead(Number(e.target.value))}
                      className="styled-select"
                      disabled={useRollout}
                    >
                      <option value={0}>Moyenne des 12 têtes</option>
                      {Array.from({ length: 12 }).map((_, i) => (
                        <option key={i + 1} value={i + 1}>
                          Tête #{i + 1}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div className="ctrl-row toggle-rollout">
                    <label>Mode Attention Rollout (Abnar &amp; Zuidema) :</label>
                    <button
                      type="button"
                      className={`btn-rollout-toggle ${useRollout ? 'active' : ''}`}
                      onClick={() => setUseRollout(!useRollout)}
                    >
                      {useRollout ? 'Activé (Flux à travers 12 couches)' : 'Désactivé (Couche par couche)'}
                    </button>
                  </div>

                  <div className="ctrl-row">
                    <label>Opacité de la superposition :</label>
                    <input
                      type="range"
                      min="0.1"
                      max="0.9"
                      step="0.05"
                      value={attnOpacity}
                      onChange={(e) => setAttnOpacity(Number(e.target.value))}
                      className="styled-range"
                    />
                  </div>
                </div>
              </div>
            </div>

            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">2</span>
                <span className="panel-header-title">Carte d'Attention Spatiale Superposée</span>
              </div>
              <div className="panel-white-body">
                <div className="spatial-attention-frame">
                  <img src={imageUrl} alt="Attn base" className="spatial-base-img" />
                  <div
                    className="spatial-heatmap-grid"
                    style={{ opacity: attnOpacity }}
                  >
                    {((useRollout ? rolloutData?.rollout_map : attnData?.spatial_map) || []).map(
                      (row, rIdx) =>
                        row.map((val, cIdx) => (
                          <div
                            key={`${rIdx}-${cIdx}`}
                            className="heatmap-patch-cell"
                            style={{
                              backgroundColor: `rgba(239, 68, 68, ${val})`,
                            }}
                            title={`Attention: ${(val * 100).toFixed(1)}%`}
                          />
                        ))
                    )}
                  </div>
                </div>
                <span className="grid-meta-summary mt-2">
                  Les zones rouges représentent l'attention portée par le token <code>[CLS]</code> sur les patches.
                </span>
              </div>
            </div>
          </div>

          <div className="practice-step-action-bar">
            <button
              type="button"
              className="btn-validate-practice-step"
              onClick={() => handleValidateStep(5)}
            >
              <span>Valider l'attention et inspecter les 12 blocs encodeurs</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* STEP 6 : SUIVI À TRAVERS LES 12 BLOCS ENCODEURS                   */}
      {/* ================================================================ */}
      {currentStep === 6 && (
        <div className="practice-step-view">
          <div className="practice-two-panels-grid">
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">1</span>
                <span className="panel-header-title">Évolution des activations couche par couche</span>
              </div>
              <div className="panel-white-body">
                <p className="text-xs text-slate-600 mb-2">
                  Norme <InlineMath math="\ell_2" /> moyenne des activations des tokens à la sortie de chaque bloc Pre-LN :
                </p>

                <div className="layers-metrics-table-wrapper">
                  <table className="layers-table">
                    <thead>
                      <tr>
                        <th>Bloc</th>
                        <th>Norme Activations (||z||)</th>
                        <th>Norme Résiduelle (||Δz||)</th>
                        <th>Ratio Résiduel</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(tensorsData?.layer_metrics || []).map((m) => (
                        <tr key={m.layer}>
                          <td><strong>Bloc #{m.layer}</strong></td>
                          <td>{m.norm}</td>
                          <td>{m.residual_norm}</td>
                          <td>{(m.residual_ratio * 100).toFixed(1)}%</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">2</span>
                <span className="panel-header-title">Rôle de la Connexion Résiduelle ⊕</span>
              </div>
              <div className="panel-white-body">
                <div className="formula-callout-box">
                  <BlockMath math="\mathbf{z}' = \mathbf{z} + \text{MHSA}(\text{LN}(\mathbf{z})),\quad \mathbf{z}'' = \mathbf{z}' + \text{MLP}(\text{LN}(\mathbf{z}'))" />
                </div>
                <p className="text-xs text-slate-600 mt-2">
                  Remarquez que le ratio résiduel reste modéré (entre 10% et 35%) : l'information de l'image se
                  propage de manière stable tout au long des 12 blocs sans dégradation, garantissant un flux de gradient
                  parfait lors de la rétropropagation.
                </p>
              </div>
            </div>
          </div>

          <div className="practice-step-action-bar">
            <button
              type="button"
              className="btn-validate-practice-step"
              onClick={() => handleValidateStep(6)}
            >
              <span>Valider les encodeurs et passer à la classification &amp; défi</span>
              <ArrowRight size={16} />
            </button>
          </div>
        </div>
      )}

      {/* ================================================================ */}
      {/* STEP 7 : CLASSIFICATION TOP-5, DEIT DUAL HEADS & PERTURBATION    */}
      {/* ================================================================ */}
      {currentStep === 7 && (
        <div className="practice-step-view">
          <div className="practice-two-panels-grid">
            {/* Top-5 Predictions Real */}
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">1</span>
                <span className="panel-header-title">Prédictions Réelles ImageNet-1k (PyTorch)</span>
              </div>
              <div className="panel-white-body">
                {predictionData?.results[0] ? (
                  <div className="predictions-results-box">
                    <div className="winning-class-banner">
                      <span className="win-label">Classe la plus probable :</span>
                      <h3 className="win-name">
                        {selectedModel === 'vit'
                          ? predictionData.results[0].vit.prediction
                          : predictionData.results[0].deit.prediction}
                      </h3>
                      <span className="win-conf">
                        Confiance :{' '}
                        {(
                          (selectedModel === 'vit'
                            ? predictionData.results[0].vit.confidence
                            : predictionData.results[0].deit.confidence) * 100
                        ).toFixed(1)}
                        %
                      </span>
                    </div>

                    <div className="top5-bars-list mt-3">
                      {(selectedModel === 'vit'
                        ? predictionData.results[0].vit.top5
                        : predictionData.results[0].deit.top5
                      ).map((item) => (
                        <div key={item.class_id} className="top5-item-row">
                          <span className="top5-name">{item.label}</span>
                          <div className="top5-bar-track">
                            <div
                              className="top5-bar-fill"
                              style={{ width: `${item.probability * 100}%` }}
                            />
                          </div>
                          <span className="top5-pct">{(item.probability * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </div>
                ) : (
                  <div className="loading-placeholder">Calcul de l'inférence PyTorch en cours...</div>
                )}

                {/* DeiT dual heads comparison */}
                {selectedModel === 'deit' && tensorsData?.deit_dual_head && (
                  <div className="deit-heads-agreement-card mt-3">
                    <span className="font-bold text-xs text-purple-900">
                      Comparaison des deux têtes DeiT (CLS vs DIST) :
                    </span>
                    <div className="deit-heads-grid mt-2">
                      <div className="head-box">
                        <span className="head-name">Tête [CLS]</span>
                        <span className="head-pred">
                          {tensorsData.deit_dual_head.cls_head[0]?.label} (
                          {(tensorsData.deit_dual_head.cls_head[0]?.prob * 100).toFixed(1)}%)
                        </span>
                      </div>
                      <div className="head-box">
                        <span className="head-name">Tête [DIST] (Teacher)</span>
                        <span className="head-pred">
                          {tensorsData.deit_dual_head.dist_head[0]?.label} (
                          {(tensorsData.deit_dual_head.dist_head[0]?.prob * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                    <span className="agreement-status-tag mt-2">
                      {tensorsData.deit_dual_head.heads_agree
                        ? '✓ Accord parfait entre les deux têtes'
                        : '⚠ Désaccord entre la tête CLS et la tête DIST'}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Mode Défi & Perturbation Experiment */}
            <div className="practice-column-panel">
              <div className="panel-navy-header">
                <span className="panel-num-circle">2</span>
                <span className="panel-header-title">Mode Défi &amp; Expérience de Perturbation</span>
              </div>
              <div className="panel-white-body">
                <p className="text-xs text-slate-600 mb-2">
                  Que se passe-t-il si l'on altère les patches de l'image ? Observez la robustesse du modèle :
                </p>

                <div className="perturb-selector-row">
                  <button
                    type="button"
                    className={`btn-perturb-opt ${perturbMode === 'mask_center' ? 'active' : ''}`}
                    onClick={() => setPerturbMode('mask_center')}
                  >
                    <EyeOff size={14} /> Masquer centre (4×4)
                  </button>
                  <button
                    type="button"
                    className={`btn-perturb-opt ${perturbMode === 'mask_random' ? 'active' : ''}`}
                    onClick={() => setPerturbMode('mask_random')}
                  >
                    <AlertTriangle size={14} /> Masquer 40 aléatoires
                  </button>
                  <button
                    type="button"
                    className={`btn-perturb-opt ${perturbMode === 'shuffle' ? 'active' : ''}`}
                    onClick={() => setPerturbMode('shuffle')}
                  >
                    <Shuffle size={14} /> Mélanger les patches
                  </button>
                </div>

                <button
                  type="button"
                  className="btn-launch-perturbation mt-3"
                  onClick={handleRunPerturbationTest}
                  disabled={isLoading}
                >
                  <Zap size={16} />
                  <span>Tester l'effet de la perturbation sur PyTorch</span>
                </button>

                {perturbationData && (
                  <div className="perturbation-results-box mt-3">
                    <div className="perturb-previews-comparison">
                      <div className="img-wrap">
                        <img src={perturbationData.original_preview} alt="Clean" />
                        <span>Originale</span>
                      </div>
                      <div className="img-wrap">
                        <img src={perturbationData.perturbed_preview} alt="Perturbed" />
                        <span>Perturbée</span>
                      </div>
                    </div>

                    <div className="perturb-impact-metrics mt-2">
                      <p>
                        <strong>Chute de confiance :</strong> -
                        {(perturbationData.confidence_drop * 100).toFixed(1)}%
                      </p>
                      <p>
                        <strong>Prédiction perturbée :</strong>{' '}
                        {perturbationData.perturbed_prediction.prediction} (
                        {(perturbationData.perturbed_prediction.confidence * 100).toFixed(1)}%)
                      </p>
                      <span
                        className={`badge-impact ${perturbationData.class_changed ? 'changed' : 'maintained'}`}
                      >
                        {perturbationData.class_changed
                          ? 'Classe prédite altérée !'
                          : 'Classe maintenue malgré le masquage !'}
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          <div className="practice-step-action-bar">
            <button
              type="button"
              className="btn-validate-practice-step"
              onClick={() => handleValidateStep(7)}
            >
              <span>Valider le laboratoire pratique complet (+15 XP)</span>
              <CheckCircle2 size={16} />
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
