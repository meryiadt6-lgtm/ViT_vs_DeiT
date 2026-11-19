import React, { useState } from 'react';
import {
  TrendingUp,
  ArrowRight,
  Layers,
  Sparkles,
  BarChart3,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';

export function ClassificationTheory() {
  const [modelType, setModelType] = useState<'vit' | 'deit'>('vit');

  return (
    <div className="theory-page">
      <div className="theory-header">
        <span className="section-pill blue">ESPACE COMPRENDRE · SECTION 07</span>
        <h1>7. Classification : Du Token [CLS] aux 1000 Probabilités</h1>
        <p className="theory-lead">
          L'aboutissement du Vision Transformer : comment la représentation vectorielle finale est extraite, normalisée et projetée vers les 1000 classes ImageNet, et comment DeiT combine les têtes CLS et DIST.
        </p>
      </div>

      <div className="theory-subtabs">
        <button
          className={`theory-tab-btn ${modelType === 'vit' ? 'active' : ''}`}
          onClick={() => setModelType('vit')}
        >
          <Layers size={15} />
          <span>Classification ViT (Tête Unique)</span>
        </button>
        <button
          className={`theory-tab-btn ${modelType === 'deit' ? 'active' : ''}`}
          onClick={() => setModelType('deit')}
        >
          <Sparkles size={15} />
          <span>Classification DeiT (Double Tête CLS + DIST)</span>
        </button>
      </div>

      <div className="two-cols-step">
        {/* Step-by-step extraction flow */}
        <div className="comp-card">
          <div className="card-top">
            <div>
              <h3>Étapes de l'Inférence de Classification</h3>
              <p>De la sortie de la couche 12 aux logits finaux.</p>
            </div>
            <span className="badge-tag">IMAGE-NET 1000 CLASSES</span>
          </div>

          <div className="steps-sublist">
            <div className="substep-item">
              <span className="step-bullet">1</span>
              <div>
                <b>Extraction du Token [CLS] (Position 0) :</b>
                <p>
                  Parmi les 197 ou 198 vecteurs en sortie de l'encodeur, nous extrayons le premier vecteur z_L^0 (768D). Les vecteurs de patches sont ignorés à cette étape.
                </p>
                <code>{"z_L^0 = Séquence[0] ∈ ℝ^768"}</code>
              </div>
            </div>

            <div className="substep-item">
              <span className="step-bullet">2</span>
              <div>
                <b>Layer Normalization Finale :</b>
                <p>
                  Le vecteur extrait passe par une normalisation finale pour stabiliser les échelles numériques avant la couche linéaire.
                </p>
                <code>{"y = LayerNorm(z_L^0) ∈ ℝ^768"}</code>
              </div>
            </div>

            <div className="substep-item">
              <span className="step-bullet">3</span>
              <div>
                <b>Projection Linéaire vers les Logits :</b>
                <p>
                  Une matrice de poids W_head (768 × 1000) et un biais b (1000) calculent les scores bruts pour les 1000 classes.
                </p>
                <code>{"Logits = y · W_head + b ∈ ℝ^1000"}</code>
              </div>
            </div>

            <div className="substep-item">
              <span className="step-bullet">4</span>
              <div>
                <b>Normalisation Softmax :</b>
                <p>
                  La fonction Softmax convertit les 1000 logits en probabilités sommant à 1 (100%).
                </p>
                <code>{"P(classe_i) = exp(Logits_i) / Σ exp(Logits_j)"}</code>
              </div>
            </div>
          </div>
        </div>

        {/* DeiT Specific Fusion / ViT Single Head */}
        <div className="comp-card">
          <div className="card-top">
            <div>
              <h3>
                {modelType === 'vit'
                  ? 'Décision Directe ViT (Single Head)'
                  : 'Fusion des Têtes DeiT (CLS Head + DIST Head)'}
              </h3>
              <p>Mécanisme de prise de décision à l'inférence.</p>
            </div>
          </div>

          {modelType === 'vit' ? (
            <div className="theory-block" style={{ padding: '16px' }}>
              <h4>Décision par le Token [CLS]</h4>
              <p style={{ margin: '12px 0', fontSize: '13px', lineHeight: '1.6' }}>
                Dans Google ViT, la décision repose à 100% sur le token CLS. La classe prédite ŷ est simplement celle qui obtient la probabilité Softmax maximale :
              </p>
              <div className="formula-box">
                <code>{"ŷ = argmax ( Softmax( W_head · LayerNorm(z_L^cls) ) )"}</code>
              </div>
              <div className="insight-pill" style={{ marginTop: '16px' }}>
                <span>✔ Simple et direct, mais dépend de la richesse sémantique accumulée au fil des 12 couches.</span>
              </div>
            </div>
          ) : (
            <div className="theory-block" style={{ padding: '16px' }}>
              <h4>Le Vote Conjoint DeiT : CLS + DIST</h4>
              <p style={{ margin: '12px 0', fontSize: '13px', lineHeight: '1.6' }}>
                DeiT extrait <strong>deux vecteurs</strong> : z_L^(cls) (position 0) et z_L^(dist) (position 1).
                Chaque vecteur passe dans sa propre tête de projection linéaire, puis leurs distributions Softmax sont moyennées :
              </p>
              <div className="formula-box highlight-box">
                <code>{"ŷ = argmax [ 0.5 · Softmax(Head_cls) + 0.5 · Softmax(Head_dist) ]"}</code>
              </div>
              <div className="insight-pill purple" style={{ marginTop: '16px' }}>
                <span>✔ Ce vote conjoint apporte un gain mesuré de <strong>+1.5% de précision Top-1</strong> sur ImageNet par rapport à une tête unique !</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Call to Practice Banner */}
      <div className="practice-cta-card">
        <div>
          <h3>Tu as compris toute la théorie ! Prêt à passer à la pratique ?</h3>
          <p>
            Ouvre l'<strong>Espace Pratiquer</strong> pour suivre les 7 étapes séquentielles guidées avec ta propre image ou nos exemples ImageNet !
          </p>
        </div>
        <NavLink to="/pratiquer" className="practice-now-btn">
          <span>Passer au Laboratoire Pratique Guidé</span>
          <ArrowRight size={16} />
        </NavLink>
      </div>
    </div>
  );
}
