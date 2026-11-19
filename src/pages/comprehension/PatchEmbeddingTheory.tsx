import React, { useState } from 'react';
import {
  Grid,
  ArrowRight,
  Maximize2,
  Sliders,
  CheckCircle2,
  Info,
  Boxes,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';

export function PatchEmbeddingTheory() {
  const [selectedPatchIdx, setSelectedPatchIdx] = useState<number>(45);
  const [patchSize, setPatchSize] = useState<number>(16);

  const numPatchesPerSide = 224 / patchSize;
  const totalPatches = numPatchesPerSide * numPatchesPerSide;
  const patchPixels = patchSize * patchSize * 3;

  const row = Math.floor(selectedPatchIdx / numPatchesPerSide);
  const col = selectedPatchIdx % numPatchesPerSide;

  return (
    <div className="theory-page">
      <div className="theory-header">
        <span className="section-pill blue">ESPACE COMPRENDRE · SECTION 03</span>
        <h1>3. Patch Embedding : Transformer l’Image en Séquence</h1>
        <p className="theory-lead">
          Un Transformer standard traite des séquences unidimensionnelles de tokens (comme des mots dans une phrase).
          Pour lui donner une image 2D, nous la divisons en une mosaïque de petites sous-régions carrées appelées <strong>Patches</strong>.
        </p>
      </div>

      <div className="two-cols-step">
        {/* Interactive Grid Card */}
        <div className="comp-card">
          <div className="card-top">
            <div>
              <h3>Division en Grille de Patches</h3>
              <p>Clique ou survole un carré pour analyser son extraction.</p>
            </div>
            <span className="badge-tag">{totalPatches} PATCHES</span>
          </div>

          <div className="patch-theory-canvas-box">
            <div className="patch-canvas-frame">
              <img
                src="/presets/samoyed.jpg"
                alt="Samoyède découpé en patches"
                className="patch-theory-img"
              />
              <div
                className="theory-grid-overlay"
                style={{
                  gridTemplateColumns: `repeat(${numPatchesPerSide}, 1fr)`,
                  gridTemplateRows: `repeat(${numPatchesPerSide}, 1fr)`,
                }}
              >
                {Array.from({ length: totalPatches }, (_, idx) => (
                  <div
                    key={idx}
                    className={`theory-patch-cell ${idx === selectedPatchIdx ? 'active' : ''}`}
                    onClick={() => setSelectedPatchIdx(idx)}
                    onMouseEnter={() => setSelectedPatchIdx(idx)}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="math-relation-card">
            <h4>Formule du Nombre de Patches :</h4>
            <div className="formula-box">
              <code>{"N = (H / P) × (W / P) = (224 / 16) × (224 / 16) = 14 × 14 = 196"}</code>
            </div>
            <div className="formula-vars">
              <span><b>H, W</b> = 224 px (hauteur et largeur de l'image)</span>
              <span><b>P</b> = 16 px (taille de chaque patch)</span>
              <span><b>N</b> = 196 patches au total</span>
            </div>
          </div>
        </div>

        {/* Patch to Vector & Linear Projection */}
        <div className="comp-card">
          <div className="card-top">
            <div>
              <h3>Du Patch 2D à la Projection Linéaire (768D)</h3>
              <p>Le processus de linéarisation et de projection dans l'espace latent.</p>
            </div>
          </div>

          {/* Selected Patch Zoom */}
          <div className="patch-inspect-box">
            <div className="inspect-row">
              <div
                className="patch-zoom-circle"
                style={{
                  backgroundImage: `url(/presets/samoyed.jpg)`,
                  backgroundSize: `${224 * (16 / patchSize) * 2}px`,
                  backgroundPosition: `-${col * 24}px -${row * 24}px`,
                }}
              />
              <div>
                <span className="badge-outline">PATCH #{selectedPatchIdx}</span>
                <h4>Patch aux Coordonnées ({row}, {col})</h4>
                <p>Zone de {patchSize} × {patchSize} pixels sur 3 canaux RVB.</p>
              </div>
            </div>

            <div className="steps-sublist">
              <div className="substep-item">
                <span className="step-bullet">1</span>
                <div>
                  <b>Aplatissement (Flattening) :</b>
                  <p>
                    Les {patchSize} × {patchSize} × 3 = <strong>{patchPixels} valeurs continues</strong> de pixels sont déroulées en un vecteur 1D x_p (768D).
                  </p>
                  <code>{"x_p ∈ ℝ^(P² · C) = ℝ^768"}</code>
                </div>
              </div>

              <div className="substep-item">
                <span className="step-bullet">2</span>
                <div>
                  <b>Projection Linéaire Apprenable (Matrice E) :</b>
                  <p>
                    Une couche Dense (équivalente à une convolution 2D de noyau 16×16 avec stride 16) projette le vecteur dans la dimension cachée du Transformer ($D=768$).
                  </p>
                  <code>{"z = x_p · E,   où E ∈ ℝ^(768 × 768)"}</code>
                </div>
              </div>

              <div className="substep-item">
                <span className="step-bullet">3</span>
                <div>
                  <b>Résultat : La Séquence d'Embeddings :</b>
                  <p>
                    Les 196 patches projetés forment une matrice bidimensionnelle de $196 \times 768$ nombres réels, prête à recevoir les tokens et encodages.
                  </p>
                  <code>{"X_embedded ∈ ℝ^(196 × 768)"}</code>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Navigation footer */}
      <div className="theory-next-bar">
        <span>Étape suivante de compréhension :</span>
        <NavLink to="/comprendre/position-tokens" className="next-theory-btn">
          <span>4. Position Embedding et Tokens Spéciaux</span>
          <ArrowRight size={15} />
        </NavLink>
      </div>
    </div>
  );
}
