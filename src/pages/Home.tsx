import React, { useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  BookOpen,
  Calculator,
  Play,
  BarChart3,
  Award,
  Crown,
  Upload,
  CheckCircle2,
  Sparkles,
  Zap,
  ArrowRight,
  RotateCcw,
  Sliders,
  Image as ImageIcon,
} from 'lucide-react';
import { useAppStore, PRESET_IMAGES } from '../store/useAppStore';
import { TeX, InlineMath } from '../components/MathFormula';

export const Home: React.FC = () => {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const {
    selectedPreset,
    imageUrl,
    imageFilename,
    preprocessMode,
    selectPreset,
    setCustomImage,
    setPreprocessMode,
    score,
    attempts,
    validatedTheorySteps,
    validatedPracticeSteps,
    badges,
    studentName,
    setStudentName,
    resetAllProgress,
  } = useAppStore();

  const theoryCount = Object.keys(validatedTheorySteps).length;
  const practiceCount = Object.keys(validatedPracticeSteps).length;
  const unlockedBadges = badges.filter((b) => b.unlocked);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.match(/^image\/(jpeg|png|webp)$/)) {
      alert('Veuillez sélectionner une image JPG, PNG ou WebP valide.');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      alert('La taille de l\'image ne doit pas dépasser 10 Mo.');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string;
      setCustomImage(dataUrl, file.name);
    };
    reader.readAsDataURL(file);
  };

  return (
    <div className="home-dashboard-container">
      {/* 1. HERO BANNER PÉDAGOGIQUE */}
      <section className="hero-studio-card">
        <div className="hero-studio-content">
          <div className="hero-kicker-pill">
            <Sparkles size={14} className="text-blue-400" />
            <span>Laboratoire Académique • Deep Learning &amp; Computer Vision</span>
          </div>
          <h1 className="hero-studio-title">
            Vision Transformer &amp; DeiT <span className="highlight-text">Learning Studio</span>
          </h1>
          <p className="hero-studio-subtitle">
            Explorez en profondeur les architectures <InlineMath math="\text{ViT}" /> (Dosovitskiy 2020) et{' '}
            <InlineMath math="\text{DeiT}" /> (Touvron 2021). De la décomposition en patches jusqu'au calcul d'attention
            multi-têtes et à la distillation avec un enseignant CNN.
          </p>

          <div className="hero-modes-row">
            <button
              className="hero-mode-btn mode-comprendre"
              onClick={() => navigate('/comprendre')}
            >
              <div className="mode-btn-icon">
                <BookOpen size={20} />
              </div>
              <div className="mode-btn-texts">
                <span className="mode-btn-title">1. Comprendre</span>
                <span className="mode-btn-desc">Schémas interactifs &amp; cours animé</span>
              </div>
              <ArrowRight size={16} className="mode-arrow" />
            </button>

            <button
              className="hero-mode-btn mode-calculer"
              onClick={() => navigate('/calculs-manuels')}
            >
              <div className="mode-btn-icon">
                <Calculator size={20} />
              </div>
              <div className="mode-btn-texts">
                <span className="mode-btn-title">2. Calculer</span>
                <span className="mode-btn-desc">Laboratoire théorique pas à pas</span>
              </div>
              <ArrowRight size={16} className="mode-arrow" />
            </button>

            <button
              className="hero-mode-btn mode-pratiquer"
              onClick={() => navigate('/pratiquer')}
            >
              <div className="mode-btn-icon">
                <Play size={20} />
              </div>
              <div className="mode-btn-texts">
                <span className="mode-btn-title">3. Pratiquer</span>
                <span className="mode-btn-desc">Vrais tenseurs PyTorch en direct</span>
              </div>
              <ArrowRight size={16} className="mode-arrow" />
            </button>
          </div>
        </div>
      </section>

      {/* 2. STATISTIQUES & GAMIFICATION APPRENANT */}
      <section className="learner-progress-banner">
        <div className="learner-profile-group">
          <div className="learner-avatar">
            <Award size={24} className="text-yellow-400" />
          </div>
          <div className="learner-info">
            <div className="learner-name-edit">
              <span className="learner-label">Étudiant(e) :</span>
              <input
                type="text"
                className="learner-name-input"
                value={studentName}
                onChange={(e) => setStudentName(e.target.value)}
                placeholder="Votre nom ou pseudonyme"
              />
            </div>
            <div className="learner-stats-pills">
              <span className="stat-pill score-pill">
                <Crown size={13} />
                <b>{score} / 100 XP</b>
              </span>
              <span className="stat-pill steps-pill">
                <CheckCircle2 size={13} />
                Théorie : <b>{theoryCount}/7</b>
              </span>
              <span className="stat-pill practice-pill">
                <Zap size={13} />
                Pratique : <b>{practiceCount}/7</b>
              </span>
              <span className="stat-pill badges-pill">
                <Sparkles size={13} />
                Badges : <b>{unlockedBadges.length}/{badges.length}</b>
              </span>
            </div>
          </div>
        </div>

        <div className="progress-actions-group">
          <button
            type="button"
            className="btn-studio-secondary"
            onClick={() => navigate('/a-propos')}
            title="Consulter vos certifications et références"
          >
            <Award size={15} />
            <span>Mon Certificat</span>
          </button>
          <button
            type="button"
            className="btn-studio-outline"
            onClick={() => {
              if (window.confirm('Réinitialiser l\'ensemble de vos scores et badges ?')) {
                resetAllProgress();
              }
            }}
            title="Réinitialiser la progression"
          >
            <RotateCcw size={14} />
            <span>Reset</span>
          </button>
        </div>
      </section>

      {/* 3. SÉLECTEUR D'IMAGE ACTIVE ET PRÉTRAITEMENT */}
      <section className="active-image-studio-panel">
        <div className="panel-header-row">
          <div className="panel-header-titles">
            <span className="panel-kicker">IMAGE D'ENTRÉE DU STUDIO</span>
            <h2>Choisissez l'image qui traversera tout le laboratoire</h2>
            <p className="panel-desc">
              Cette image est automatiquement synchronisée avec les pages <strong>Calculs manuels</strong>,{' '}
              <strong>Pratiquer</strong> et <strong>Comparaison</strong>.
            </p>
          </div>

          <div className="preprocess-toggle-box">
            <span className="preprocess-label">
              <Sliders size={14} /> Pré-traitement :
            </span>
            <div className="preprocess-options">
              <button
                type="button"
                className={`preprocess-chip ${preprocessMode === 'resize_crop_256' ? 'active' : ''}`}
                onClick={() => setPreprocessMode('resize_crop_256')}
                title="Redimensionnement à 256x256 puis cadrage central 224x224 (Standard ImageNet)"
              >
                Resize 256 → Crop 224
              </button>
              <button
                type="button"
                className={`preprocess-chip ${preprocessMode === 'direct_224' ? 'active' : ''}`}
                onClick={() => setPreprocessMode('direct_224')}
                title="Redimensionnement direct sans découpage"
              >
                Resize direct 224×224
              </button>
            </div>
          </div>
        </div>

        <div className="image-selection-grid">
          {/* Preset Images Gallery */}
          <div className="presets-gallery-col">
            <span className="gallery-section-label">Galerie d'exemples académiques :</span>
            <div className="presets-cards-flex">
              {PRESET_IMAGES.map((preset) => {
                const isSelected = selectedPreset === preset.id;
                return (
                  <button
                    key={preset.id}
                    type="button"
                    className={`preset-card-item ${isSelected ? 'active-card' : ''}`}
                    onClick={() => selectPreset(preset.id)}
                  >
                    <div className="preset-thumb-frame">
                      <img src={preset.url} alt={preset.name} className="preset-thumb-img" />
                      {isSelected && (
                        <div className="preset-active-check">
                          <CheckCircle2 size={16} />
                        </div>
                      )}
                    </div>
                    <div className="preset-card-meta">
                      <span className="preset-title">{preset.name}</span>
                      <span className="preset-class-label">{preset.trueClass}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          {/* Active Preview & Custom Upload */}
          <div className="active-preview-col">
            <span className="gallery-section-label">Image active &amp; Téléversement :</span>
            <div className="active-preview-card">
              <div className="active-img-wrapper">
                <img src={imageUrl} alt="Active preview" className="active-main-preview-img" />
                <div className="active-img-specs-badge">
                  <span>224 × 224 px</span>
                  <span>•</span>
                  <span>RGB (3 canaux)</span>
                </div>
              </div>

              <div className="upload-controls-box">
                <input
                  type="file"
                  ref={fileInputRef}
                  className="hidden-file-input"
                  accept="image/jpeg,image/png,image/webp"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
                <button
                  type="button"
                  className="btn-upload-custom"
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload size={16} />
                  <span>Importer votre image (JPG/PNG)</span>
                </button>
                <span className="upload-limits-note">
                  Max 10 Mo • Format carré ou rectangle (recadré automatiquement à 224×224)
                </span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 4. COMPARATIF SYNTHÉTIQUE VIT VS DEIT */}
      <section className="vit-deit-spotlight-cards">
        <div className="spotlight-card vit-card">
          <div className="spotlight-badge vit-tag">ARCHITECTURE ORIGINALE (2020)</div>
          <h3>Google Vision Transformer (ViT-Base)</h3>
          <p className="spotlight-intro">
            Premier modèle pur Transformer appliqué à la vision sans aucune convolution, traitant les patches
            d'images exactement comme les tokens de mots dans le NLP.
          </p>
          <ul className="spotlight-specs-list">
            <li>
              <span>Patches d'entrée :</span> <b>196 patches (16 × 16 px)</b>
            </li>
            <li>
              <span>Dimension latente <InlineMath math="D" /> :</span> <b>768 dimensions</b>
            </li>
            <li>
              <span>Couches d'encodeur :</span> <b>12 blocs Transformer Pre-LN</b>
            </li>
            <li>
              <span>Têtes d'attention :</span> <b>12 têtes (<InlineMath math="d_k = 64" />)</b>
            </li>
            <li>
              <span>Séquence totale :</span> <b>197 tokens (196 + 1 [CLS])</b>
            </li>
            <li>
              <span>Tête de classification :</span> <b>Linéaire sur le token [CLS]</b>
            </li>
          </ul>
        </div>

        <div className="spotlight-card deit-card">
          <div className="spotlight-badge deit-tag">DISTILLATION PAR ATTENTION (2021)</div>
          <h3>Meta Data-efficient Image Transformer (DeiT-Base)</h3>
          <p className="spotlight-intro">
            Introduit un mécanisme de <strong>distillation par attention</strong> guidé par un réseau CNN enseignant
            (RegNetY), permettant d'atteindre des performances records sans base de données géante (JFT-300M).
          </p>
          <ul className="spotlight-specs-list">
            <li>
              <span>Patches d'entrée :</span> <b>196 patches (16 × 16 px)</b>
            </li>
            <li>
              <span>Dimension latente <InlineMath math="D" /> :</span> <b>768 dimensions</b>
            </li>
            <li>
              <span>Couches d'encodeur :</span> <b>12 blocs Transformer Pre-LN</b>
            </li>
            <li>
              <span>Token spécial :</span> <b>Token de distillation [DIST]</b>
            </li>
            <li>
              <span>Séquence totale :</span> <b>198 tokens (196 + [CLS] + [DIST])</b>
            </li>
            <li>
              <span>Prédiction finale :</span> <b>Moyenne des deux têtes (CLS &amp; DIST)</b>
            </li>
          </ul>
        </div>
      </section>
    </div>
  );
};

