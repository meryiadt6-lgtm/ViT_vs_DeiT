import React, { useState } from 'react';
import {
  Boxes,
  ArrowRight,
  Sparkles,
  Layers,
  HelpCircle,
  Sliders,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';

export function PositionalTokensTheory() {
  const [selectedTokenTab, setSelectedTokenTab] = useState<'all' | 'cls' | 'dist' | 'pos'>('all');

  return (
    <div className="theory-page">
      <div className="theory-header">
        <span className="section-pill blue">ESPACE COMPRENDRE · SECTION 04</span>
        <h1>4. Position Embedding et Tokens Spéciaux</h1>
        <p className="theory-lead">
          Comment les tokens spéciaux (<code>[CLS]</code> et <code>[DIST]</code>) sont insérés dans la séquence, et pourquoi les <strong>Position Embeddings</strong> sont essentiels pour que le Transformer ne perde pas l'organisation spatiale 2D.
        </p>
      </div>

      {/* Token Family Cards */}
      <div className="theory-subtabs">
        <button
          className={`theory-tab-btn ${selectedTokenTab === 'all' ? 'active' : ''}`}
          onClick={() => setSelectedTokenTab('all')}
        >
          <Boxes size={15} />
          <span>Vue d'Ensemble de la Séquence</span>
        </button>
        <button
          className={`theory-tab-btn ${selectedTokenTab === 'cls' ? 'active' : ''}`}
          onClick={() => setSelectedTokenTab('cls')}
        >
          <Layers size={15} />
          <span>Le Token [CLS]</span>
        </button>
        <button
          className={`theory-tab-btn ${selectedTokenTab === 'dist' ? 'active' : ''}`}
          onClick={() => setSelectedTokenTab('dist')}
        >
          <Sparkles size={15} />
          <span>Le Token [DIST] (DeiT)</span>
        </button>
        <button
          className={`theory-tab-btn ${selectedTokenTab === 'pos' ? 'active' : ''}`}
          onClick={() => setSelectedTokenTab('pos')}
        >
          <HelpCircle size={15} />
          <span>Pourquoi les Position Embeddings ?</span>
        </button>
      </div>

      {/* Visual Sequence Assembly Diagram */}
      <div className="comp-card">
        <div className="card-top">
          <div>
            <h3>La Concaténation des Tokens et l'Addition Positionnelle</h3>
            <p>Structure de la séquence d'entrée complète $z_0$ injectée dans le premier bloc Transformer.</p>
          </div>
          <span className="badge-tag">DIMENSION : [1, 198, 768] (DEIT) / [1, 197, 768] (VIT)</span>
        </div>

        <div className="sequence-assembly-visual">
          <div className="assembly-row">
            {/* [CLS] Token */}
            <div className="token-block-item cls-box">
              <span className="pos-indicator">Position 0</span>
              <b>[CLS]</b>
              <small>Classification Token (768D)</small>
            </div>

            {/* [DIST] Token */}
            <div className="token-block-item dist-box">
              <span className="pos-indicator">Position 1 (DeiT)</span>
              <b>[DIST]</b>
              <small>Distillation Token (768D)</small>
            </div>

            {/* Patch tokens 1 to 196 */}
            <div className="token-block-item patch-box">
              <span className="pos-indicator">Positions 2 .. 197</span>
              <b>196 Patch Tokens</b>
              <small>Vecteurs projetés des régions 16×16 px</small>
            </div>
          </div>

          <div className="assembly-plus-row">
            <span className="plus-symbol">+</span>
            <b>Addition Élément par Élément des Position Embeddings 1D :</b>
          </div>

          <div className="assembly-row pos-row">
            <div className="token-block-item pos-box">
              <span className="pos-indicator">E_pos[0]</span>
              <b>Pos 0 Embedding</b>
            </div>
            <div className="token-block-item pos-box">
              <span className="pos-indicator">E_pos[1]</span>
              <b>Pos 1 Embedding</b>
            </div>
            <div className="token-block-item pos-box">
              <span className="pos-indicator">E_pos[2 .. 197]</span>
              <b>196 Position Embeddings (1D Apprenables)</b>
            </div>
          </div>

          <div className="assembly-equal-row">
            <span className="equal-symbol">=</span>
            <div className="final-sequence-banner">
              <b>Séquence Entrée Z₀ prête pour l'Encodeur :</b>
              <code>{"z_0 = [x_cls ; x_dist ; x_p^1 E ; ... ; x_p^196 E] + E_pos"}</code>
            </div>
          </div>
        </div>
      </div>

      {/* Explanatory cards */}
      <div className="two-cols-step">
        <div className="comp-card">
          <div className="card-top">
            <div className="icon-title">
              <Layers className="text-blue" size={20} />
              <h4>Le Rôle du Token [CLS]</h4>
            </div>
          </div>
          <p className="theory-text">
            Le token <code>[CLS]</code> est un vecteur apprenable indépendant de l'image inséré en tête de séquence.
            Comme l'opération de Self-Attention est symétrique et sans notion spatiale native, le token [CLS] échange de l'information avec chacun des 196 patches au fil des 12 couches.
            À la sortie de la couche 12, son vecteur condense une représentation globale de l'image et sert d'entrée à la tête de classification.
          </p>
        </div>

        <div className="comp-card">
          <div className="card-top">
            <div className="icon-title">
              <Sparkles className="text-purple" size={20} />
              <h4>Le Rôle du Token [DIST] (Spécifique à DeiT)</h4>
            </div>
          </div>
          <p className="theory-text">
            Introduit par Touvron et al. (Meta AI), le token <code>[DIST]</code> agit comme un deuxième récepteur spécialisé.
            Pendant l'entraînement, il est supervisé pour reproduire spécifiquement la décision prédite par le Teacher ConvNet (RegNet).
            À l'inférence, il vote aux côtés du token [CLS], apportant le biais inductif hérité de la convolution.
          </p>
        </div>
      </div>

      {/* Why are Position Embeddings Necessary? */}
      <div className="comp-card">
        <div className="card-top">
          <div className="icon-title">
            <Info className="text-blue" size={20} />
            <h4>Pourquoi les Position Embeddings sont-ils indispensables ?</h4>
          </div>
          <span className="badge-tag">INVARIANCE PAR PERMUTATION</span>
        </div>

        <div className="scale-explanation">
          <p>
            Le mécanisme d'attention est <strong>invariant par permutation</strong> : si vous mélangez aléatoirement l'ordre des patches dans la séquence sans leur ajouter d'encodage de position, la formule de Self-Attention produira exactement le même résultat !
          </p>
          <div className="formula-box">
            <code>{"Attention(P · Q,  P · K,  P · V) = P · Attention(Q, K, V)"}</code>
          </div>
          <p>
            Le Transformer ne sait pas naturellement distinguer le coin supérieur gauche du coin inférieur droit.
            L'ajout des vecteurs E_pos (appris lors de l'entraînement) fournit à chaque position de token une "signature" unique qui rétablit la topologie géométrique 2D de l'image.
          </p>
        </div>
      </div>

      {/* Navigation footer */}
      <div className="theory-next-bar">
        <span>Étape suivante de compréhension :</span>
        <NavLink to="/comprendre/self-attention" className="next-theory-btn">
          <span>5. Mécanisme de Self-Attention ($Q, K, V$)</span>
          <ArrowRight size={15} />
        </NavLink>
      </div>
    </div>
  );
}
