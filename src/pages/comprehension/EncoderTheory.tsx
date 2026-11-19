import React, { useState } from 'react';
import {
  Layers,
  ChevronRight,
  ArrowRight,
  Sliders,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';

export function EncoderTheory() {
  const [activeOp, setActiveOp] = useState<string>('ln1');

  const blockComponents = [
    {
      id: 'ln1',
      title: '1. Layer Normalization 1 (Pre-LN)',
      math: 'x₁ = LayerNorm(z_{l-1})',
      desc: 'Normalise chaque token indépendamment (moyenne 0, variance 1) AVANT d’entrer dans le module d’attention. Cette disposition Pre-LN est cruciale pour la stabilité de l’entraînement en profondeur.',
      badge: 'STABILITÉ'
    },
    {
      id: 'msa',
      title: '2. Multi-Head Self-Attention (12 Têtes)',
      math: 'y₁ = MSA(x₁) = Concat(head₁ .. head₁₂) · W_O',
      desc: '12 têtes opèrent en parallèle sur des sous-espaces de 64 dimensions (12 × 64 = 768), captant simultanément relations locales de texture et relations sémantiques globales.',
      badge: 'COMMUNICATION'
    },
    {
      id: 'res1',
      title: '3. Première Connexion Résiduelle (Skip Connection)',
      math: 'z\'_l = z_{l-1} + y₁',
      desc: 'Additionne directement le vecteur d’entrée z_{l-1} à la sortie du module d’attention. Crée une autoroute sans obstruction pour la propagation du gradient.',
      badge: 'FLUX DE GRADIENT'
    },
    {
      id: 'ln2',
      title: '4. Layer Normalization 2 (Pre-LN)',
      math: 'x₂ = LayerNorm(z\'_l)',
      desc: 'Deuxième normalisation par couches appliquée avant le réseau feed-forward pour stabiliser les activations avant l’expansion non-linéaire.',
      badge: 'STABILITÉ'
    },
    {
      id: 'mlp',
      title: '5. MLP / Feed Forward Network (Expansion 4× + GELU)',
      math: 'y₂ = W₂ · GELU(W₁ · x₂ + b₁) + b₂',
      desc: 'Deux couches denses avec facteur d’expansion 4 : projection de 768 vers 3072 dimensions, activation non-linéaire GELU, puis reprojection vers 768 dimensions.',
      badge: 'TRANSFORMATION'
    },
    {
      id: 'res2',
      title: '6. Deuxième Connexion Résiduelle',
      math: 'z_l = z\'_l + y₂',
      desc: 'Addition finale de la sortie du MLP avec z\'_l. Le vecteur z_l ainsi enrichi est transmis au bloc Transformer suivant ou à la tête de classification.',
      badge: 'SORTIE DU BLOC'
    }
  ];

  const current = blockComponents.find((c) => c.id === activeOp) || blockComponents[0];

  return (
    <div className="theory-page">
      <div className="theory-header">
        <span className="section-pill blue">ESPACE COMPRENDRE · SECTION 06</span>
        <h1>6. Transformer Encoder : Le Bloc Pre-LN en Détail</h1>
        <p className="theory-lead">
          L'encodeur est constitué de 12 blocs identiques empilés. Chaque bloc applique une alternance rigoureuse de <strong>Multi-Head Self-Attention</strong>, de <strong>connexions résiduelles</strong> et de <strong>MLP à activation GELU</strong>.
        </p>
      </div>

      <div className="two-cols-step">
        {/* Left: Interactive list of block components */}
        <div className="comp-card">
          <div className="card-top">
            <div>
              <h3>Schéma du Bloc Transformer (Pre-LN)</h3>
              <p>Clique sur chaque composant pour examiner son équation et son rôle.</p>
            </div>
            <span className="badge-tag">6 ÉTAPES INTERNES</span>
          </div>

          <div className="ops-pipeline-flow">
            {blockComponents.map((c) => (
              <div
                key={c.id}
                className={`op-flow-item ${activeOp === c.id ? 'active' : ''}`}
                onClick={() => setActiveOp(c.id)}
              >
                <div className="op-left">
                  <div>
                    <div className="op-name">{c.title}</div>
                    <code className="op-math-preview">{c.math}</code>
                  </div>
                </div>
                <div className="op-right">
                  <span className="op-tag">{c.badge}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right: Detailed focus card */}
        <div className="comp-card">
          <div className="card-top">
            <div>
              <span className="badge-outline">{current.badge}</span>
              <h3 style={{ marginTop: '8px' }}>{current.title}</h3>
            </div>
          </div>

          <div className="inspect-formula-display">
            <code>{current.math}</code>
          </div>

          <p className="theory-text" style={{ fontSize: '13px', lineHeight: '1.6', margin: '14px 0' }}>
            {current.desc}
          </p>

          <div className="specs-table" style={{ marginTop: '18px' }}>
            <div>
              <span>Dimension d'Entrée :</span>
              <b>[1, 197, 768] (ViT) ou [1, 198, 768] (DeiT)</b>
            </div>
            <div>
              <span>Dimension Sortie :</span>
              <b>Identique à l'entrée (Isotrope)</b>
            </div>
            <div>
              <span>Nombre total de blocs :</span>
              <b>12 Blocs empilés en série</b>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation footer */}
      <div className="theory-next-bar">
        <span>Étape suivante de compréhension :</span>
        <NavLink to="/comprendre/classification" className="next-theory-btn">
          <span>7. Têtes de Classification & Décision</span>
          <ArrowRight size={15} />
        </NavLink>
      </div>
    </div>
  );
}
