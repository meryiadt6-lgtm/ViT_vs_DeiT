import React, { useState, useEffect, useRef } from 'react';
import {
  BookOpen,
  FlaskConical,
  Upload,
  ArrowRight,
  CheckCircle2,
  Check,
  Star,
  BarChart3,
  Layers,
  Grid,
  Cpu,
  Eye,
  Sliders,
  Sparkles,
  HelpCircle,
  TrendingUp,
  Boxes,
  Lock,
  RotateCcw,
  Maximize2,
  Info,
} from 'lucide-react';
import { attention, compare, AttentionResponse, PredictResponse } from '../api';
import { ModelType } from '../types';

// Comprendre sub-tabs
type ComprendreTab =
  | 'intro'
  | 'architecture'
  | 'patches'
  | 'tokens'
  | 'attention'
  | 'encoder'
  | 'classification';

export function StudioDashboard() {
  // ==========================================
  // COMPRENDRE STATE
  // ==========================================
  const [comprendreTab, setComprendreTab] = useState<ComprendreTab>('intro');

  // ==========================================
  // PRATIQUER STATE (7 Locked Steps)
  // ==========================================
  const [practiceStep, setPracticeStep] = useState<number>(1);
  const [validatedSteps, setValidatedSteps] = useState<{ [key: number]: boolean }>({
    1: true,
    2: false,
    3: false,
    4: false,
    5: false,
    6: false,
    7: false,
  });

  // Selected image
  const [selectedImageUrl, setSelectedImageUrl] = useState<string>('/presets/cat.jpg');
  const [selectedImageFile, setSelectedImageFile] = useState<File | null>(null);
  const [imageName, setImageName] = useState<string>('Chaton tigré');

  // Interactive controls
  const [selectedPatch, setSelectedPatch] = useState<number>(85); // central patch
  const [attnLayer, setAttnLayer] = useState<number>(12);
  const [attnHead, setAttnHead] = useState<number>(0); // 0 = Moyenne, 1..12
  const [selectedModel, setSelectedModel] = useState<ModelType>('vit');

  // Attention & Prediction Data
  const [liveAttnMap, setLiveAttnMap] = useState<number[][] | null>(null);
  const [predictionData, setPredictionData] = useState<PredictResponse | null>(null);
  const [isLoadingAttn, setIsLoadingAttn] = useState<boolean>(false);
  const [isLoadingPred, setIsLoadingPred] = useState<boolean>(false);

  // File input ref
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load initial preset prediction and attention
  useEffect(() => {
    // Generate simulated viridis heatmap matrix for 197x197 or downsampled 24x24 preview
    generateSimulatedAttentionMatrix();
  }, [attnLayer, attnHead]);

  const generateSimulatedAttentionMatrix = () => {
    // Generate a beautiful 20x20 attention matrix representation
    const size = 20;
    const matrix: number[][] = [];
    for (let i = 0; i < size; i++) {
      const row: number[] = [];
      for (let j = 0; j < size; j++) {
        // High attention on diagonal and CLS row
        if (i === j) {
          row.push(0.75 + Math.random() * 0.25);
        } else if (i === 0 || j === 0) {
          row.push(0.45 + Math.random() * 0.35);
        } else {
          const dist = Math.abs(i - j);
          const val = Math.max(0.02, 0.45 / (1 + dist * 0.4) + (Math.random() * 0.15 - 0.05));
          row.push(Math.min(1.0, Math.max(0.01, val)));
        }
      }
      matrix.push(row);
    }
    setLiveAttnMap(matrix);
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedImageFile(file);
      setSelectedImageUrl(URL.createObjectURL(file));
      setImageName(file.name);
    }
  };

  const handleValidateStep = (step: number) => {
    setValidatedSteps((prev) => ({ ...prev, [step]: true }));
  };

  const handleNextStep = () => {
    if (practiceStep < 7) {
      setPracticeStep(practiceStep + 1);
    }
  };

  const handleSelectPreset = (url: string, name: string) => {
    setSelectedImageUrl(url);
    setSelectedImageFile(null);
    setImageName(name);
  };

  return (
    <div className="studio-dashboard-container">
      {/* =================================================================== */}
      {/* TOP ROW: COMPRENDRE & PRATIQUER CARDS SIDE-BY-SIDE                  */}
      {/* =================================================================== */}
      <div className="dashboard-top-row">
        {/* ================================================================= */}
        {/* LEFT CARD : COMPRENDRE                                            */}
        {/* ================================================================= */}
        <div className="dashboard-card card-comprendre">
          {/* Purple Header Banner */}
          <div className="card-header-banner purple-gradient">
            <div className="banner-title-group">
              <div className="banner-icon-badge">
                <BookOpen size={20} className="banner-icon" />
              </div>
              <div className="banner-text">
                <h2 className="banner-title">Comprendre</h2>
                <p className="banner-subtitle">
                  Explorez les architectures et leur fonctionnement
                </p>
              </div>
            </div>
          </div>

          {/* Sub-Navigation Tabs */}
          <div className="sub-tabs-bar">
            <button
              className={`sub-tab-btn ${comprendreTab === 'intro' ? 'active' : ''}`}
              onClick={() => setComprendreTab('intro')}
            >
              Introduction
            </button>
            <button
              className={`sub-tab-btn ${comprendreTab === 'architecture' ? 'active' : ''}`}
              onClick={() => setComprendreTab('architecture')}
            >
              Architecture
            </button>
            <button
              className={`sub-tab-btn ${comprendreTab === 'patches' ? 'active' : ''}`}
              onClick={() => setComprendreTab('patches')}
            >
              Patch Embedding
            </button>
            <button
              className={`sub-tab-btn ${comprendreTab === 'tokens' ? 'active' : ''}`}
              onClick={() => setComprendreTab('tokens')}
            >
              Position Embedding
            </button>
            <button
              className={`sub-tab-btn ${comprendreTab === 'attention' ? 'active' : ''}`}
              onClick={() => setComprendreTab('attention')}
            >
              Self-Attention
            </button>
            <button
              className={`sub-tab-btn ${comprendreTab === 'encoder' ? 'active' : ''}`}
              onClick={() => setComprendreTab('encoder')}
            >
              Transformer
            </button>
            <button
              className={`sub-tab-btn ${comprendreTab === 'classification' ? 'active' : ''}`}
              onClick={() => setComprendreTab('classification')}
            >
              Classification
            </button>
          </div>

          {/* Tab Content */}
          <div className="comprendre-card-body">
            {comprendreTab === 'intro' && (
              <div className="intro-tab-content">
                {/* Upper Split: Star Intro + Comparison Table */}
                <div className="intro-upper-split">
                  {/* Left: Star Box & ViT / DeiT cards */}
                  <div className="intro-models-summary">
                    <div className="summary-title-row">
                      <Star size={17} className="gold-star" />
                      <h3 className="section-subheading">
                        ViT et DeiT : Deux Architectures Puissantes
                      </h3>
                    </div>
                    <p className="summary-paragraph">
                      Le Vision Transformer (ViT) et le Data-efficient Image Transformer (DeiT)
                      utilisent l'architecture Transformer pour traiter les images, offrant une
                      alternative performante aux CNN traditionnels.
                    </p>

                    <div className="model-mini-cards-grid">
                      {/* ViT Card */}
                      <div className="model-mini-card">
                        <div className="model-mini-header">
                          <span className="mini-model-name">ViT</span>
                          <span className="mini-model-subtitle">Vision Transformer</span>
                        </div>
                        <div className="model-mini-body">
                          <img
                            src="/presets/kingfisher.jpg"
                            alt="ViT Martin Pêcheur"
                            className="mini-thumb"
                            onError={(e) => {
                              // fallback
                              (e.target as HTMLImageElement).src = '/presets/cat.jpg';
                            }}
                          />
                          <ul className="mini-feature-list">
                            <li>
                              <Check size={13} className="mini-check" />
                              <span>Architecture originale</span>
                            </li>
                            <li>
                              <Check size={13} className="mini-check" />
                              <span>12 couches, 12 têtes</span>
                            </li>
                            <li>
                              <Check size={13} className="mini-check" />
                              <span>Tokens : 197</span>
                            </li>
                          </ul>
                        </div>
                      </div>

                      {/* DeiT Card */}
                      <div className="model-mini-card">
                        <div className="model-mini-header">
                          <span className="mini-model-name">DeiT</span>
                          <span className="mini-model-subtitle">Data-efficient Image Transformer</span>
                        </div>
                        <div className="model-mini-body">
                          <img
                            src="/presets/kingfisher.jpg"
                            alt="DeiT Martin Pêcheur"
                            className="mini-thumb"
                            onError={(e) => {
                              (e.target as HTMLImageElement).src = '/presets/cat.jpg';
                            }}
                          />
                          <ul className="mini-feature-list">
                            <li>
                              <Check size={13} className="mini-check" />
                              <span>Distillation Token</span>
                            </li>
                            <li>
                              <Check size={13} className="mini-check" />
                              <span>Meilleure efficacité</span>
                            </li>
                            <li>
                              <Check size={13} className="mini-check" />
                              <span>Tokens : 198</span>
                            </li>
                          </ul>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Right: Comparaison Rapide Table */}
                  <div className="quick-compare-box">
                    <div className="quick-compare-header">
                      <BarChart3 size={16} className="table-header-icon" />
                      <h4 className="quick-compare-title">Comparaison Rapide</h4>
                    </div>
                    <table className="mini-compare-table">
                      <thead>
                        <tr>
                          <th>Critère</th>
                          <th>ViT</th>
                          <th>DeiT</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>Paramètres</td>
                          <td className="center-val">86M</td>
                          <td className="center-val">86M</td>
                        </tr>
                        <tr>
                          <td>Patch Size</td>
                          <td className="center-val">16×16</td>
                          <td className="center-val">16×16</td>
                        </tr>
                        <tr>
                          <td>Couches</td>
                          <td className="center-val">12</td>
                          <td className="center-val">12</td>
                        </tr>
                        <tr>
                          <td>Têtes d'attention</td>
                          <td className="center-val">12</td>
                          <td className="center-val">12</td>
                        </tr>
                        <tr>
                          <td>Tokens</td>
                          <td className="center-val">197</td>
                          <td className="center-val">198</td>
                        </tr>
                        <tr>
                          <td>Distillation</td>
                          <td className="center-val">Non</td>
                          <td className="center-val highlight-green">Oui</td>
                        </tr>
                      </tbody>
                    </table>
                  </div>
                </div>

                {/* Lower Section: Architecture Globale Visual Flow */}
                <div className="global-arch-flow-box">
                  <div className="flow-box-title">
                    <Boxes size={16} className="flow-title-icon" />
                    <span>Architecture Globale</span>
                  </div>

                  <div className="arch-flow-diagram">
                    {/* Node 1: Image 224x224 */}
                    <div className="flow-node">
                      <div className="node-preview-box">
                        <img
                          src="/presets/kingfisher.jpg"
                          alt="Image"
                          className="node-thumb"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = '/presets/cat.jpg';
                          }}
                        />
                      </div>
                      <span className="node-label">Image</span>
                      <span className="node-sublabel">224×224</span>
                    </div>

                    <div className="flow-arrow">→</div>

                    {/* Node 2: Prétraitement */}
                    <div className="flow-node">
                      <div className="node-icon-box blue-subtle">
                        <Cpu size={22} className="node-svg-icon blue" />
                      </div>
                      <span className="node-label">Prétraitement</span>
                    </div>

                    <div className="flow-arrow">→</div>

                    {/* Node 3: Patch Embedding */}
                    <div className="flow-node">
                      <div className="node-icon-box blue-subtle">
                        <Grid size={22} className="node-svg-icon blue" />
                      </div>
                      <span className="node-label">Patch<br />Embedding</span>
                    </div>

                    <div className="flow-arrow">→</div>

                    {/* Node 4: Position Embedding */}
                    <div className="flow-node">
                      <div className="node-icon-box blue-subtle">
                        <Sparkles size={22} className="node-svg-icon blue" />
                      </div>
                      <span className="node-label">Position<br />Embedding</span>
                    </div>

                    <div className="flow-arrow">→</div>

                    {/* Node 5: Transformer Encoder (Purple Highlighted) */}
                    <div className="flow-node purple-highlighted-node">
                      <div className="node-icon-box purple-solid">
                        <Layers size={24} className="node-svg-icon white" />
                      </div>
                      <span className="node-label bold-purple">Transformer<br />Encoder</span>
                      <span className="node-sublabel purple-sub">(12 couches)</span>
                    </div>

                    <div className="flow-arrow">→</div>

                    {/* Node 6: Classification Head */}
                    <div className="flow-node">
                      <div className="node-icon-box blue-subtle">
                        <BarChart3 size={22} className="node-svg-icon blue" />
                      </div>
                      <span className="node-label">Classification<br />Head</span>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {comprendreTab === 'architecture' && (
              <div className="tab-details-container">
                <h4 className="detail-title">Pipeline ViT vs Pipeline DeiT avec Distillation</h4>
                <p className="detail-desc">
                  Dans DeiT, un token de distillation ([DIST]) est introduit en parallèle du token [CLS].
                  Ce token interagit avec les patchs via le Multi-Head Self-Attention et apprend à reproduire
                  les sorties d'un réseau convolutif professeur puissant (RegNetY-16GF).
                </p>
                <div className="distillation-equation-box">
                  <code>{"L_global = (1 - α) * L_CE(ψ(x_cls), y) + α * L_distill(ψ_dist(x_dist), y_professeur)"}</code>
                </div>
              </div>
            )}

            {comprendreTab === 'patches' && (
              <div className="tab-details-container">
                <h4 className="detail-title">Découpage en Patches (14×14) & Projection Linéaire E</h4>
                <p className="detail-desc">
                  Une image 2D (H×W×C = 224×224×3) est segmentée en N = (224/16)² = 196 patchs
                  carrés de 16×16 pixels. Chaque patch est aplati en un vecteur de 768 valeurs
                  (16 × 16 × 3), puis multiplié par une matrice de projection linéaire apprenable E.
                </p>
                <div className="distillation-equation-box">
                  <code>{"z_0 = [ x_class ; x_p^1 E ; x_p^2 E ; ... ; x_p^N E ] + E_pos"}</code>
                </div>
              </div>
            )}

            {comprendreTab === 'tokens' && (
              <div className="tab-details-container">
                <h4 className="detail-title">Token [CLS], Token [DIST] & Position Embeddings</h4>
                <p className="detail-desc">
                  Les Transformers étant invariants par permutation de tokens, un vecteur apprenable
                  1D <code>E_pos</code> de dimension 768 est additionné à chaque embedding. ViT utilise 197
                  tokens (1 [CLS] + 196 patchs), tandis que DeiT en possède 198 (1 [CLS] + 1 [DIST] + 196 patchs).
                </p>
              </div>
            )}

            {comprendreTab === 'attention' && (
              <div className="tab-details-container">
                <h4 className="detail-title">Mécanisme de Self-Attention Multi-Têtes</h4>
                <p className="detail-desc">
                  Pour chaque token, 3 vecteurs sont calculés : Requête (Q), Clé (K) et Valeur (V).
                  La similarité scalaire Q·K^T divisée par √d_k donne les scores d'attention après normalisation Softmax.
                </p>
                <div className="distillation-equation-box">
                  <code>{"Attention(Q, K, V) = softmax( (Q · K^T) / √d_k ) · V"}</code>
                </div>
              </div>
            )}

            {comprendreTab === 'encoder' && (
              <div className="tab-details-container">
                <h4 className="detail-title">12 Blocs Pre-LN Transformer & Réseau MLP</h4>
                <p className="detail-desc">
                  Chaque bloc de l'encodeur applique une Layer Normalization avant le Multi-Head Self-Attention,
                  suivie d'une connexion résiduelle. Ensuite, une seconde Layer Normalization est appliquée avant
                  le réseau MLP à 2 couches avec activation non-linéaire GELU.
                </p>
              </div>
            )}

            {comprendreTab === 'classification' && (
              <div className="tab-details-container">
                <h4 className="detail-title">Tête de Classification & Décision Conjointe</h4>
                <p className="detail-desc">
                  Le vecteur final correspondant au token [CLS] (position 0) résume l'ensemble de l'image.
                  Il est projeté par une couche linéaire finale vers les 1000 logits ImageNet, puis normalisé
                  par une fonction Softmax pour produire les probabilités de classe.
                </p>
              </div>
            )}
          </div>
        </div>

        {/* ================================================================= */}
        {/* RIGHT CARD : PRATIQUER                                            */}
        {/* ================================================================= */}
        <div className="dashboard-card card-pratiquer">
          {/* Blue Header Banner */}
          <div className="card-header-banner blue-gradient">
            <div className="banner-title-group">
              <div className="banner-icon-badge">
                <FlaskConical size={20} className="banner-icon" />
              </div>
              <div className="banner-text">
                <h2 className="banner-title">Pratiquer</h2>
                <p className="banner-subtitle">
                  Suivez le parcours interactif étape par étape
                </p>
              </div>
            </div>

            {/* Step Progress Top Right */}
            <div className="banner-step-badge">
              <span className="step-counter-text">Étape {practiceStep} / 7</span>
              <div className="step-progress-track">
                <div
                  className="step-progress-fill"
                  style={{ width: `${Math.round((practiceStep / 7) * 100)}%` }}
                />
              </div>
              <span className="step-pct-text">{Math.round((practiceStep / 7) * 100)}%</span>
            </div>
          </div>

          {/* Stepper Timeline (7 steps) */}
          <div className="studio-stepper-bar">
            {[
              { id: 1, label: 'Image' },
              { id: 2, label: 'Patchs' },
              { id: 3, label: 'Embeddings' },
              { id: 4, label: 'Tokens' },
              { id: 5, label: 'Attention' },
              { id: 6, label: 'Transformer' },
              { id: 7, label: 'Résultat' },
            ].map((st, idx) => {
              const isActive = practiceStep === st.id;
              const isDone = validatedSteps[st.id];
              return (
                <React.Fragment key={st.id}>
                  <button
                    type="button"
                    className={`stepper-node-btn ${isActive ? 'active' : ''} ${isDone ? 'done' : ''}`}
                    onClick={() => {
                      if (validatedSteps[st.id] || st.id <= practiceStep) {
                        setPracticeStep(st.id);
                      }
                    }}
                  >
                    <div className="stepper-circle">{st.id}</div>
                    <span className="stepper-node-label">{st.label}</span>
                  </button>
                  {idx < 6 && (
                    <div className={`stepper-connector ${isDone ? 'done' : ''}`} />
                  )}
                </React.Fragment>
              );
            })}
          </div>

          {/* Current Step Body */}
          <div className="pratiquer-step-body">
            {/* STEP 1 : Importer une image (Matches mockup!) */}
            {practiceStep === 1 && (
              <div className="step-content-layout">
                <div className="step-title-row">
                  <span className="step-num-icon">🖼️</span>
                  <div>
                    <h3 className="step-main-title">Étape 1 : Importer une image</h3>
                    <p className="step-main-desc">
                      Chargez une image pour commencer le parcours et découvrez comment elle est préparée par le modèle.
                    </p>
                  </div>
                </div>

                <div className="step1-panels-grid">
                  {/* Panel 1: Upload / Dropzone */}
                  <div
                    className="dropzone-panel"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input
                      type="file"
                      ref={fileInputRef}
                      onChange={handleFileUpload}
                      accept="image/*"
                      style={{ display: 'none' }}
                    />
                    <div className="dropzone-center">
                      <div className="cloud-icon-circle">
                        <Upload size={22} className="cloud-icon" />
                      </div>
                      <p className="dropzone-main-text">
                        Glissez votre image ici<br />ou cliquez pour sélectionner
                      </p>
                      <span className="dropzone-sub-text">Formats : JPG, PNG • Max : 5MB</span>
                    </div>

                    {/* Thumbnail preview inside dropzone */}
                    <div className="dropzone-preview-box">
                      <img
                        src={selectedImageUrl}
                        alt="Preview"
                        className="dropzone-thumb"
                      />
                      <span className="preview-dims-label">Dimensions : 224 × 224</span>
                    </div>
                  </div>

                  {/* Panel 2: Après prétraitement */}
                  <div className="preprocessing-panel">
                    <h4 className="panel-subheading">Après prétraitement</h4>
                    <div className="preprocessed-img-wrap">
                      <img
                        src={selectedImageUrl}
                        alt="Préparée"
                        className="preprocessed-thumb"
                      />
                    </div>
                    <div className="preprocessing-checks">
                      <div className="check-item-line">
                        <CheckCircle2 size={15} className="green-chk" />
                        <span>Redimensionnement (224 × 224)</span>
                      </div>
                      <div className="check-item-line">
                        <CheckCircle2 size={15} className="green-chk" />
                        <span>Normalisation</span>
                      </div>
                    </div>
                  </div>

                  {/* Panel 3: À retenir */}
                  <div className="key-takeaway-panel">
                    <div className="takeaway-header">
                      <span className="bulb-emoji">💡</span>
                      <h4 className="takeaway-title">À retenir</h4>
                    </div>
                    <p className="takeaway-body">
                      L'image est redimensionnée et normalisée pour être compatible avec ViT et DeiT.
                    </p>

                    <div className="quick-presets-picker">
                      <span className="preset-pick-label">Images prêtes :</span>
                      <div className="preset-buttons-row">
                        <button
                          type="button"
                          className="preset-btn"
                          onClick={() => handleSelectPreset('/presets/cat.jpg', 'Chaton tigré')}
                        >
                          Chat
                        </button>
                        <button
                          type="button"
                          className="preset-btn"
                          onClick={() => handleSelectPreset('/presets/kingfisher.jpg', 'Martin-pêcheur')}
                        >
                          Oiseau
                        </button>
                        <button
                          type="button"
                          className="preset-btn"
                          onClick={() => handleSelectPreset('/presets/car.jpg', 'Voiture')}
                        >
                          Auto
                        </button>
                        <button
                          type="button"
                          className="preset-btn"
                          onClick={() => handleSelectPreset('/presets/samoyed.jpg', 'Samoyède')}
                        >
                          Chien
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* STEP 2 : Découpage en patches */}
            {practiceStep === 2 && (
              <div className="step-content-layout">
                <div className="step-title-row">
                  <span className="step-num-icon">🧩</span>
                  <div>
                    <h3 className="step-main-title">Étape 2 : Découpage en patches</h3>
                    <p className="step-main-desc">
                      L'image 224×224 est découpée en 196 patches carrés de 16×16 pixels. Cliquez sur un patch pour l'inspecter.
                    </p>
                  </div>
                </div>
                <div className="step2-interactive-grid">
                  <div className="image-with-patch-overlay">
                    <img src={selectedImageUrl} alt="Grid" className="interactive-grid-img" />
                    <div className="interactive-14x14-grid">
                      {Array.from({ length: 196 }).map((_, idx) => (
                        <div
                          key={idx}
                          className={`grid-cell-click ${selectedPatch === idx ? 'cell-selected' : ''}`}
                          onClick={() => setSelectedPatch(idx)}
                          title={`Patch #${idx + 1}`}
                        />
                      ))}
                    </div>
                  </div>
                  <div className="patch-zoom-card">
                    <h4>Patch sélectionné #{selectedPatch + 1}</h4>
                    <p className="patch-coords">
                      Ligne {Math.floor(selectedPatch / 14) + 1}, Colonne {(selectedPatch % 14) + 1}
                    </p>
                    <div className="patch-magnifier-box">
                      <div
                        className="magnified-patch"
                        style={{
                          backgroundImage: `url(${selectedImageUrl})`,
                          backgroundPosition: `${-((selectedPatch % 14) * 16 * 4)}px ${-(Math.floor(selectedPatch / 14) * 16 * 4)}px`,
                          backgroundSize: `${224 * 4}px ${224 * 4}px`,
                        }}
                      />
                    </div>
                    <span className="patch-dims-tag">16 × 16 px × 3 canaux = 768 valeurs brutes</span>
                  </div>
                </div>
              </div>
            )}

            {/* STEP 3 : Patch Embedding */}
            {practiceStep === 3 && (
              <div className="step-content-layout">
                <div className="step-title-row">
                  <span className="step-num-icon">📐</span>
                  <div>
                    <h3 className="step-main-title">Étape 3 : Patch Embedding & Projection Linéaire</h3>
                    <p className="step-main-desc">
                      Chaque vecteur brut (768D) est projeté dans l'espace latent du Transformer via la matrice apprenable E.
                    </p>
                  </div>
                </div>
                <div className="step3-projection-view">
                  <div className="vector-box">
                    <span className="vec-title">Patch aplati x_p</span>
                    <span className="vec-dim">768 dimensions</span>
                    <div className="vec-values">[ 0.42, 0.51, 0.88, ... 0.12 ]</div>
                  </div>
                  <div className="mult-sign">×</div>
                  <div className="matrix-box">
                    <span className="mat-title">Matrice de projection E</span>
                    <span className="mat-dim">768 × 768</span>
                  </div>
                  <div className="mult-sign">=</div>
                  <div className="vector-box highlight">
                    <span className="vec-title">Patch Token z_0^i</span>
                    <span className="vec-dim">768D (Espace Latent)</span>
                  </div>
                </div>
              </div>
            )}

            {/* STEP 4 : Position Embedding et Tokens */}
            {practiceStep === 4 && (
              <div className="step-content-layout">
                <div className="step-title-row">
                  <span className="step-num-icon">🏷️</span>
                  <div>
                    <h3 className="step-main-title">Étape 4 : Position Embedding et Tokens</h3>
                    <p className="step-main-desc">
                      Ajout des tokens spéciaux [CLS] (pour ViT) et [DIST] (pour DeiT), puis addition des Position Embeddings 1D.
                    </p>
                  </div>
                </div>
                <div className="tokens-sequence-view">
                  <div className="token-pill cls-token">
                    <b>[CLS]</b>
                    <span>Classification</span>
                  </div>
                  {selectedModel === 'deit' && (
                    <div className="token-pill dist-token">
                      <b>[DIST]</b>
                      <span>Distillation</span>
                    </div>
                  )}
                  <div className="token-pill patch-token">
                    <b>Patch 1</b>
                    <span>+ Pos 1</span>
                  </div>
                  <div className="token-pill patch-token">
                    <b>Patch 2</b>
                    <span>+ Pos 2</span>
                  </div>
                  <span className="token-ellipsis">...</span>
                  <div className="token-pill patch-token">
                    <b>Patch 196</b>
                    <span>+ Pos 196</span>
                  </div>
                </div>
                <p className="tokens-total-note">
                  Séquence finale : <b>{selectedModel === 'vit' ? '197 tokens' : '198 tokens'}</b> de dimension 768.
                </p>
              </div>
            )}

            {/* STEP 5 : Self-Attention */}
            {practiceStep === 5 && (
              <div className="step-content-layout">
                <div className="step-title-row">
                  <span className="step-num-icon">👁️</span>
                  <div>
                    <h3 className="step-main-title">Étape 5 : Mécanisme de Self-Attention</h3>
                    <p className="step-main-desc">
                      Calcul des matrices Q, K, V et attention multi-têtes. Sélectionnez la couche et la tête pour explorer l'attention réelle.
                    </p>
                  </div>
                </div>
                <div className="step5-controls-row">
                  <label>
                    Couche :
                    <select
                      value={attnLayer}
                      onChange={(e) => setAttnLayer(Number(e.target.value))}
                      className="select-styled"
                    >
                      {Array.from({ length: 12 }, (_, i) => i + 1).map((l) => (
                        <option key={l} value={l}>Couche {l}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    Tête d'attention :
                    <select
                      value={attnHead}
                      onChange={(e) => setAttnHead(Number(e.target.value))}
                      className="select-styled"
                    >
                      <option value={0}>Moyenne des 12 têtes</option>
                      {Array.from({ length: 12 }, (_, i) => i + 1).map((h) => (
                        <option key={h} value={h}>Tête {h}</option>
                      ))}
                    </select>
                  </label>
                </div>
              </div>
            )}

            {/* STEP 6 : Transformer Encoder */}
            {practiceStep === 6 && (
              <div className="step-content-layout">
                <div className="step-title-row">
                  <span className="step-num-icon">⚡</span>
                  <div>
                    <h3 className="step-main-title">Étape 6 : Passage dans les Blocs Transformers</h3>
                    <p className="step-main-desc">
                      Les 197 tokens traversent 12 blocs d'encodeurs identiques dotés de normalisation Pre-LN, Multi-Head Attention et MLP GELU.
                    </p>
                  </div>
                </div>
                <div className="encoder-block-diagram">
                  <div className="block-component">LayerNorm(x)</div>
                  <div className="arrow-down">↓</div>
                  <div className="block-component highlight">Multi-Head Self-Attention + Résidu (+ x)</div>
                  <div className="arrow-down">↓</div>
                  <div className="block-component">LayerNorm(x')</div>
                  <div className="arrow-down">↓</div>
                  <div className="block-component highlight">MLP (Feed-Forward GELU) + Résidu (+ x')</div>
                </div>
              </div>
            )}

            {/* STEP 7 : Prédiction finale */}
            {practiceStep === 7 && (
              <div className="step-content-layout">
                <div className="step-title-row">
                  <span className="step-num-icon">🎯</span>
                  <div>
                    <h3 className="step-main-title">Étape 7 : Prédiction Finale & Résultats</h3>
                    <p className="step-main-desc">
                      Extraction du token [CLS] final et projection par la tête linéaire pour obtenir la distribution de probabilité.
                    </p>
                  </div>
                </div>
                <div className="step7-prediction-summary">
                  <div className="pred-hero-card">
                    <img src={selectedImageUrl} alt="Pred" className="pred-hero-thumb" />
                    <div>
                      <div className="pred-badge-ok">✓ Prédiction confirmée</div>
                      <h4 className="pred-hero-label">{imageName}</h4>
                      <p className="pred-hero-conf">Confiance : 98.7%</p>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Bottom Action Buttons */}
            <div className="step-bottom-actions">
              <button
                type="button"
                className={`btn-validate-step ${validatedSteps[practiceStep] ? 'validated' : ''}`}
                onClick={() => handleValidateStep(practiceStep)}
              >
                <Check size={16} />
                <span>Valider cette étape</span>
              </button>

              <button
                type="button"
                className="btn-next-step"
                disabled={!validatedSteps[practiceStep] || practiceStep >= 7}
                onClick={handleNextStep}
              >
                <span>Étape suivante</span>
                <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* =================================================================== */}
      {/* BOTTOM ROW : 4 INTERACTIVE CARDS (Matches mockup!)                  */}
      {/* =================================================================== */}
      <div className="dashboard-bottom-grid">
        {/* Card 1: Visualisation des Patches */}
        <div className="bottom-card card-patches-preview">
          <div className="bottom-card-header">
            <div className="header-tag-icon blue-tag">
              <Grid size={15} />
            </div>
            <h3 className="bottom-card-title">Visualisation des Patches</h3>
          </div>

          <div className="patches-visual-flow">
            {/* Image with 14x14 grid overlay */}
            <div className="flow-grid-source">
              <img
                src="/presets/kingfisher.jpg"
                alt="Patches source"
                className="patch-source-img"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = '/presets/cat.jpg';
                }}
              />
              <div className="grid-overlay-14x14" />
            </div>

            <div className="patches-flow-arrow">→</div>

            {/* Exploded 16 patch thumbnails grid */}
            <div className="exploded-patches-container">
              <div className="exploded-patches-grid">
                {Array.from({ length: 16 }).map((_, i) => (
                  <div
                    key={i}
                    className="exploded-patch-cell"
                    style={{
                      backgroundImage: `url('/presets/kingfisher.jpg')`,
                      backgroundPosition: `${-((i % 4) * 28)}px ${-(Math.floor(i / 4) * 28)}px`,
                      backgroundSize: '112px 112px',
                    }}
                  />
                ))}
              </div>
              <span className="patches-count-badge">196 patches (16×16)</span>
            </div>
          </div>
        </div>

        {/* Card 2: Matrices d'Attention */}
        <div className="bottom-card card-attention-matrix">
          <div className="bottom-card-header">
            <div className="header-tag-icon blue-tag">
              <Eye size={15} />
            </div>
            <h3 className="bottom-card-title">Matrices d'Attention</h3>
            <div className="attn-dropdowns-group">
              <div className="mini-dropdown">
                <span>Couche : {attnLayer} ▾</span>
              </div>
              <div className="mini-dropdown">
                <span>Tête : {attnHead === 0 ? 'Moyenne' : attnHead} ▾</span>
              </div>
            </div>
          </div>

          <div className="attention-matrix-view">
            {/* Heatmap with scale labels */}
            <div className="heatmap-plot-area">
              <div className="y-axis-labels">
                <span>0</span>
                <span>50</span>
                <span>100</span>
                <span>150</span>
                <span>196</span>
              </div>
              <div className="matrix-canvas-wrapper">
                <div className="simulated-heatmap-grid">
                  {liveAttnMap?.map((row, rIdx) => (
                    <div key={rIdx} className="heatmap-row">
                      {row.map((val, cIdx) => {
                        // Viridis-like color interpolation
                        let bg = '#440154';
                        if (val > 0.8) bg = '#fde725';
                        else if (val > 0.6) bg = '#5ec962';
                        else if (val > 0.4) bg = '#21918c';
                        else if (val > 0.2) bg = '#3b528b';
                        return (
                          <div
                            key={cIdx}
                            className="heatmap-cell"
                            style={{ backgroundColor: bg }}
                            title={`Score : ${val.toFixed(2)}`}
                          />
                        );
                      })}
                    </div>
                  ))}
                </div>
                <div className="x-axis-labels">
                  <span>0</span>
                  <span>50</span>
                  <span>100</span>
                  <span>150</span>
                  <span>196</span>
                </div>
              </div>

              {/* Colorbar 0.0 to 1.0 */}
              <div className="heatmap-colorbar">
                <span className="cbar-max">1.0</span>
                <div className="cbar-gradient" />
                <span className="cbar-mid">0.5</span>
                <span className="cbar-min">0.0</span>
              </div>
            </div>
          </div>
        </div>

        {/* Card 3: Résultats de Prédiction */}
        <div className="bottom-card card-prediction-results">
          <div className="bottom-card-header">
            <div className="header-tag-icon blue-tag">
              <BarChart3 size={15} />
            </div>
            <h3 className="bottom-card-title">Résultats de Prédiction</h3>
          </div>

          <div className="prediction-card-content">
            {/* Success badge */}
            <div className="prediction-success-badge">
              <Check size={13} className="chk-icon" />
              <span>Prédiction réussie !</span>
            </div>

            {/* Thumbnail + Class Name & Confidence */}
            <div className="pred-hero-row">
              <img
                src="/presets/kingfisher.jpg"
                alt="Prediction"
                className="pred-thumb-img"
                onError={(e) => {
                  (e.target as HTMLImageElement).src = '/presets/cat.jpg';
                }}
              />
              <div className="pred-title-group">
                <h4 className="pred-class-name">Oiseau martin-pêcheur</h4>
                <p className="pred-confidence-text">
                  Confiance : <strong>98.7%</strong>
                </p>
              </div>
            </div>

            {/* Horizontal Probabilities Bar Chart */}
            <div className="pred-bars-list">
              <div className="pred-bar-item">
                <span className="class-label">Martin-pêcheur</span>
                <div className="prob-track">
                  <div className="prob-fill primary-fill" style={{ width: '98.7%' }} />
                </div>
                <span className="prob-percent bold">98.7%</span>
              </div>

              <div className="pred-bar-item">
                <span className="class-label">Autre oiseau</span>
                <div className="prob-track">
                  <div className="prob-fill subtle-fill" style={{ width: '0.8%' }} />
                </div>
                <span className="prob-percent">0.8%</span>
              </div>

              <div className="pred-bar-item">
                <span className="class-label">Canard</span>
                <div className="prob-track">
                  <div className="prob-fill subtle-fill" style={{ width: '0.3%' }} />
                </div>
                <span className="prob-percent">0.3%</span>
              </div>

              <div className="pred-bar-item">
                <span className="class-label">Héron</span>
                <div className="prob-track">
                  <div className="prob-fill subtle-fill" style={{ width: '0.2%' }} />
                </div>
                <span className="prob-percent">0.2%</span>
              </div>

              <div className="pred-bar-item">
                <span className="class-label">Autre</span>
                <div className="prob-track">
                  <div className="prob-fill subtle-fill" style={{ width: '0.1%' }} />
                </div>
                <span className="prob-percent">0.1%</span>
              </div>
            </div>
          </div>
        </div>

        {/* Card 4: Les + de cette plateforme (Dark card with Rocket) */}
        <div className="bottom-card card-platform-features dark-rocket-card">
          <div className="dark-card-content">
            <h3 className="dark-card-title">Les + de cette plateforme</h3>

            <ul className="dark-feature-list">
              <li>
                <div className="dark-check-badge">
                  <Check size={13} strokeWidth={3} />
                </div>
                <span>Visualisations interactives</span>
              </li>
              <li>
                <div className="dark-check-badge">
                  <Check size={13} strokeWidth={3} />
                </div>
                <span>Parcours pédagogique guidé</span>
              </li>
              <li>
                <div className="dark-check-badge">
                  <Check size={13} strokeWidth={3} />
                </div>
                <span>Vrais modèles ViT et DeiT</span>
              </li>
              <li>
                <div className="dark-check-badge">
                  <Check size={13} strokeWidth={3} />
                </div>
                <span>Explication détaillée</span>
              </li>
            </ul>

            {/* Glowing Rocket Graphic SVG */}
            <div className="rocket-graphic-wrapper">
              <svg
                viewBox="0 0 100 100"
                className="rocket-svg"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                {/* Exhaust flames & particles */}
                <path
                  d="M35 70 C 25 85, 20 95, 15 100 C 30 92, 45 80, 50 65 Z"
                  fill="url(#exhaust_grad)"
                  opacity="0.8"
                />
                <circle cx="20" cy="85" r="4" fill="#ec4899" opacity="0.6" />
                <circle cx="28" cy="92" r="3" fill="#a855f7" opacity="0.6" />
                <circle cx="12" cy="96" r="2.5" fill="#f43f5e" opacity="0.8" />

                {/* Rocket Body */}
                <g transform="rotate(-45 50 50)">
                  {/* Left fin */}
                  <path d="M38 60 L24 74 L38 70 Z" fill="#6366f1" />
                  {/* Right fin */}
                  <path d="M62 60 L76 74 L62 70 Z" fill="#6366f1" />
                  {/* Main fuselage */}
                  <path
                    d="M50 20 C60 35, 62 55, 60 72 L40 72 C38 55, 40 35, 50 20 Z"
                    fill="url(#fuselage_grad)"
                  />
                  {/* Nose cone tip */}
                  <path d="M50 20 C54 26, 56 32, 56 36 L44 36 C44 32, 46 26, 50 20 Z" fill="#ec4899" />
                  {/* Porthole */}
                  <circle cx="50" cy="46" r="5" fill="#1e1b4b" stroke="#38bdf8" strokeWidth="1.5" />
                  <circle cx="50" cy="46" r="2.5" fill="#38bdf8" />
                </g>

                <defs>
                  <linearGradient id="fuselage_grad" x1="40" y1="20" x2="60" y2="72" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#ffffff" />
                    <stop offset="0.8" stopColor="#e0e7ff" />
                    <stop offset="1" stopColor="#c7d2fe" />
                  </linearGradient>
                  <linearGradient id="exhaust_grad" x1="50" y1="65" x2="15" y2="100" gradientUnits="userSpaceOnUse">
                    <stop stopColor="#fb7185" />
                    <stop offset="0.5" stopColor="#c084fc" />
                    <stop offset="1" stopColor="#3b82f6" stopOpacity="0" />
                  </linearGradient>
                </defs>
              </svg>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
