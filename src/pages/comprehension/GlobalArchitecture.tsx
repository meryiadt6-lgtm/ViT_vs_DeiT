import React, { useState } from 'react';
import {
  ArrowRight,
  Layers,
  Sparkles,
  Database,
  Boxes,
  Cpu,
  BrainCircuit,
  Eye,
  CheckCircle2,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';

export function GlobalArchitecture() {
  const [pipelineModel, setPipelineModel] = useState<'vit' | 'deit'>('vit');

  return (
    <div className="theory-page">
      <div className="theory-header">
        <span className="section-pill blue">ESPACE COMPRENDRE · SECTION 02</span>
        <h1>2. Architecture Globale : Le Pipeline Complet</h1>
        <p className="theory-lead">
          Le flux complet d’une image de son format brut RVB (224×224×3) jusqu’à la distribution des probabilités finales de 1000 classes ImageNet.
        </p>
      </div>

      {/* Model toggle for pipeline */}
      <div className="theory-subtabs">
        <button
          className={`theory-tab-btn ${pipelineModel === 'vit' ? 'active' : ''}`}
          onClick={() => setPipelineModel('vit')}
        >
          <Layers size={15} />
          <span>Pipeline ViT (Vision Transformer)</span>
        </button>
        <button
          className={`theory-tab-btn ${pipelineModel === 'deit' ? 'active' : ''}`}
          onClick={() => setPipelineModel('deit')}
        >
          <Sparkles size={15} />
          <span>Pipeline DeiT (avec Distillation Token & Teacher)</span>
        </button>
      </div>

      {/* Main interactive pipeline card */}
      <div className="comp-card">
        <div className="card-top">
          <div>
            <h3>
              {pipelineModel === 'vit'
                ? 'Pipeline Standard ViT : Image → Prétraitement → Patch Embedding → Position Embedding → Transformer Encoder → Classification Head'
                : 'Pipeline DeiT avec Distillation : Token [DIST], Guidance du Teacher CNN et Fusion à l’Inférence'}
            </h3>
            <p>Visualise chaque étape de la transformation des données.</p>
          </div>
          <span className="badge-tag">
            {pipelineModel === 'vit' ? 'ViT-BASE (197 TOKENS)' : 'DEIT-BASE DISTILLED (198 TOKENS)'}
          </span>
        </div>

        {/* Visual Pipeline Stages */}
        <div className="pipeline-flow-vertical">
          {/* Etape 1 : Image & Prétraitement */}
          <div className="pipeline-stage-item">
            <div className="stage-num">01</div>
            <div className="stage-card">
              <div className="stage-head">
                <span className="stage-badge">ENTRÉE VISUELLE</span>
                <h4>Image & Prétraitement Officiel</h4>
              </div>
              <p>
                L'image est redimensionnée en $224 \times 224$ pixels, convertie en valeurs continues $[0, 1]$, puis centrée et réduite avec les moyennes ImageNet ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).
              </p>
              <div className="tensor-shape-tag">
                <span>Forme Tensorielle :</span>
                <code>torch.Size([1, 3, 224, 224])</code>
              </div>
            </div>
          </div>

          <div className="pipeline-arrow-separator">
            <ArrowRight size={20} className="down-arrow" />
          </div>

          {/* Etape 2 : Patch Embedding */}
          <div className="pipeline-stage-item">
            <div className="stage-num">02</div>
            <div className="stage-card">
              <div className="stage-head">
                <span className="stage-badge">DÉCOUPAGE & PROJECTION</span>
                <h4>Patch Partitioning & Patch Embedding ($E$)</h4>
              </div>
              <p>
                L'image est découpée en N = 14 × 14 = 196 patches de 16 × 16 × 3 = 768 pixels.
                Chaque patch aplati x_p est multiplié par une matrice de projection linéaire apprenable E (768 × 768).
              </p>
              <div className="tensor-shape-tag">
                <span>Forme Tensorielle :</span>
                <code>torch.Size([1, 196, 768])</code>
              </div>
            </div>
          </div>

          <div className="pipeline-arrow-separator">
            <ArrowRight size={20} className="down-arrow" />
          </div>

          {/* Etape 3 : Position Embedding et Tokens */}
          <div className="pipeline-stage-item">
            <div className="stage-num">03</div>
            <div className="stage-card highlight-stage">
              <div className="stage-head">
                <span className="stage-badge">{pipelineModel === 'vit' ? '1 TOKEN SPÉCIAL' : '2 TOKENS SPÉCIAUX'}</span>
                <h4>Insertion des Tokens & Position Embedding 1D (E_pos)</h4>
              </div>
              <p>
                {pipelineModel === 'vit' ? (
                  <>
                    Le token de classification <code>[CLS]</code> est concaténé à l'indice 0 (séquence de 197 tokens).
                    Une table d'encodages positionnels 1D apprenables E_pos (197 × 768) est additionnée élément par élément.
                  </>
                ) : (
                  <>
                    Deux tokens spéciaux sont insérés : <code>[CLS]</code> à l'indice 0 et <code>[DIST]</code> (Distillation Token) à l'indice 1 (séquence de 198 tokens).
                    La table E_pos (198 × 768) est additionnée pour encoder les positions relatives.
                  </>
                )}
              </p>
              <div className="tensor-shape-tag">
                <span>Forme Séquence Z₀ :</span>
                <code>{pipelineModel === 'vit' ? 'torch.Size([1, 197, 768])' : 'torch.Size([1, 198, 768])'}</code>
              </div>
            </div>
          </div>

          <div className="pipeline-arrow-separator">
            <ArrowRight size={20} className="down-arrow" />
          </div>

          {/* Etape 4 : Transformer Encoder */}
          <div className="pipeline-stage-item">
            <div className="stage-num">04</div>
            <div className="stage-card">
              <div className="stage-head">
                <span className="stage-badge">CORE TRANSFORMER</span>
                <h4>Encodeur Transformer : 12 Blocs Pre-LN Identiques</h4>
              </div>
              <p>
                La séquence traverse 12 blocs successifs. À chaque bloc :
                LayerNorm 1 $\to$ Multi-Head Self-Attention (12 têtes de 64D) $\to$ Connexion résiduelle $\to$
                LayerNorm 2 $\to$ MLP Feed-Forward (expansion 3072D avec activation GELU) $\to$ Connexion résiduelle.
              </p>
              <div className="tensor-shape-tag">
                <span>Forme Sortie Couche 12 :</span>
                <code>{pipelineModel === 'vit' ? 'torch.Size([1, 197, 768])' : 'torch.Size([1, 198, 768])'}</code>
              </div>
            </div>
          </div>

          <div className="pipeline-arrow-separator">
            <ArrowRight size={20} className="down-arrow" />
          </div>

          {/* Etape 5 : Classification Head */}
          <div className="pipeline-stage-item">
            <div className="stage-num">05</div>
            <div className="stage-card highlight-final">
              <div className="stage-head">
                <span className="stage-badge">DÉCISION FINALE</span>
                <h4>
                  {pipelineModel === 'vit'
                    ? 'Tête de Classification MLP sur [CLS]'
                    : 'Double Tête de Classification (CLS Head + DIST Head)'}
                </h4>
              </div>
              <p>
                {pipelineModel === 'vit' ? (
                  <>
                    Le vecteur final $z_L^0$ du token <code>[CLS]</code> est extrait, normalisé par un LayerNorm final, puis projeté vers les 1000 logits ImageNet via une couche linéaire.
                  </>
                ) : (
                  <>
                    Le vecteur du token <code>[CLS]</code> passe dans la tête de classification standard, et le vecteur de <code>[DIST]</code> passe dans la tête de distillation.
                    À l'inférence, leurs probabilités Softmax sont moyennées : ŷ = 0.5 × (P_cls + P_dist).
                  </>
                )}
              </p>
              <div className="tensor-shape-tag">
                <span>Sortie Logits :</span>
                <code>torch.Size([1, 1000])</code>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Special Box for DeiT Distillation Mechanism */}
      {pipelineModel === 'deit' && (
        <div className="comp-card deit-detail-card">
          <div className="card-top">
            <div className="icon-title">
              <BrainCircuit className="text-purple" size={22} />
              <h4>Le Mécanisme de Distillation Spécifique à DeiT</h4>
            </div>
            <span className="badge-tag deit-tag">TEACHER → STUDENT</span>
          </div>

          <div className="two-cols-comp">
            <div className="theory-block">
              <b>Rôle du Teacher ConvNet (RegNetY-16GF) :</b>
              <p>
                Pendant l'entraînement, un puissant réseau de neurones convolutif pré-entraîné sert de guide.
                Il génère pour chaque image une prédiction y_teacher riche en corrélations spatiales et en biais inductif.
              </p>
            </div>
            <div className="theory-block">
              <b>Rôle du Distillation Token [DIST] :</b>
              <p>
                Le token [DIST] est injecté aux côtés des patches et du token [CLS].
                Grâce au Self-Attention, il communique avec tous les patches et apprend spécifiquement à reproduire la décision du Teacher par Cross-Entropy (Hard Distillation).
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Navigation footer */}
      <div className="theory-next-bar">
        <span>Étape suivante de compréhension :</span>
        <NavLink to="/comprendre/patch-embedding" className="next-theory-btn">
          <span>3. Patch Embedding en Détail</span>
          <ArrowRight size={15} />
        </NavLink>
      </div>
    </div>
  );
}
