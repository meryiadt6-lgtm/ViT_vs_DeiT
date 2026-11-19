import React, { useState, useEffect } from 'react';
import {
  Eye,
  ArrowRight,
  Sliders,
  Sparkles,
  Info,
  Layers,
  CheckCircle2,
  AlertCircle,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';
import { attention, AttentionResponse } from '../../api';

export function SelfAttentionTheory() {
  const [layer, setLayer] = useState<number>(12);
  const [head, setHead] = useState<number>(0);
  const [model, setModel] = useState<string>('vit');
  const [attentionData, setAttentionData] = useState<AttentionResponse | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string>('');

  // Sample toy 4x4 matrix for theoretical exploration
  const [hoveredCell, setHoveredCell] = useState<{ q: number; k: number } | null>(null);
  const toyTokens = ['[CLS]', 'Patch #1 (Tête)', 'Patch #45 (Yeux)', 'Patch #90 (Museau)'];
  const toyAttentionMatrix = [
    [0.10, 0.22, 0.45, 0.23],
    [0.15, 0.48, 0.22, 0.15],
    [0.11, 0.18, 0.47, 0.24],
    [0.08, 0.12, 0.36, 0.44],
  ];

  // Try fetching real attention on mount using the preset samoyed image
  useEffect(() => {
    fetchRealAttention();
  }, [layer, head, model]);

  const fetchRealAttention = async () => {
    setIsLoading(true);
    setErrorMsg('');
    try {
      const res = await fetch('/presets/samoyed.jpg');
      const blob = await res.blob();
      const file = new File([blob], 'samoyed.jpg', { type: 'image/jpeg' });
      const data = await attention(file, model, layer, head);
      setAttentionData(data);
    } catch (err: any) {
      setErrorMsg(err.message || 'Impossible de contacter l’API locale');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="theory-page">
      <div className="theory-header">
        <span className="section-pill blue">ESPACE COMPRENDRE · SECTION 05</span>
        <h1>5. Self-Attention : Le Mécanisme de Communication Globale</h1>
        <p className="theory-lead">
          Comment les vecteurs <strong>Query</strong>, <strong>Key</strong> et <strong>Value</strong> permettent à chaque token de peser son affinité avec tous les autres pour agréger une représentation contextuelle riche.
        </p>
      </div>

      {/* Core Equation Box */}
      <div className="comp-card attention-formula-card">
        <div className="card-top">
          <div>
            <h3>La Formule du Scaled Dot-Product Attention</h3>
            <p>La règle mathématique universelle qui relie les tokens.</p>
          </div>
          <span className="badge-tag">VASWANI ET AL. / DOSOVITSKIY ET AL.</span>
        </div>

        <div className="giant-equation-box">
          <code>{"Attention(Q, K, V) = Softmax( (Q · Kᵀ) / √d_k ) · V"}</code>
        </div>

        <div className="qkv-breakdown-grid">
          <div className="qkv-card q-card">
            <div className="qkv-tag">QUERY (Q)</div>
            <b>"Ce que je recherche"</b>
            <p>Vecteur émis par le token demandeur décrivant ce qu'il recherche dans les autres régions.</p>
            <code>{"Q = X · W_Q ∈ ℝ^(N × d_k)"}</code>
          </div>

          <div className="qkv-card k-card">
            <div className="qkv-tag">KEY (K)</div>
            <b>"Ce que je possède"</b>
            <p>Vecteur émis par chaque token pour indexer ses propres propriétés visuelles.</p>
            <code>{"K = X · W_K ∈ ℝ^(N × d_k)"}</code>
          </div>

          <div className="qkv-card v-card">
            <div className="qkv-tag">VALUE (V)</div>
            <b>"L'information transmise"</b>
            <p>Le contenu sémantique réel qui sera transmis si le produit scalaire Q · K est élevé.</p>
            <code>{"V = X · W_V ∈ ℝ^(N × d_k)"}</code>
          </div>
        </div>
      </div>

      {/* Interactive Controls to inspect Layer and Head on real model */}
      <div className="comp-card">
        <div className="card-top">
          <div>
            <h3>Visualisation des Attentions Réelles du Modèle</h3>
            <p>Choisis le modèle, la couche Transformer (1-12) et la tête d'attention spécifique (1-12 ou moyenne).</p>
          </div>
          <span className="badge-outline">POIDS RÉELS PYTORCH</span>
        </div>

        <div className="attention-controls-bar">
          <div className="control-group">
            <label>Modèle :</label>
            <select value={model} onChange={(e) => setModel(e.target.value)}>
              <option value="vit">Google ViT-Base (197 Tokens)</option>
              <option value="deit">Meta DeiT-Base Distilled (198 Tokens)</option>
            </select>
          </div>

          <div className="control-group">
            <label>Couche :</label>
            <select value={layer} onChange={(e) => setLayer(parseInt(e.target.value))}>
              {Array.from({ length: 12 }, (_, i) => (
                <option key={i + 1} value={i + 1}>
                  Couche {i + 1} {i === 0 ? '(Basses fréquences)' : i === 11 ? '(Sémantique globale)' : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="control-group">
            <label>Tête d'Attention :</label>
            <select value={head} onChange={(e) => setHead(parseInt(e.target.value))}>
              <option value={0}>Moyenne des 12 Têtes</option>
              {Array.from({ length: 12 }, (_, i) => (
                <option key={i + 1} value={i + 1}>
                  Tête #{i + 1}
                </option>
              ))}
            </select>
          </div>
        </div>

        {errorMsg && (
          <div className="error-alert">
            <AlertCircle size={18} />
            <div>
              <b>Information API :</b> {errorMsg} (Vérifie que le backend FastAPI est lancé sur le port 8000).
            </div>
          </div>
        )}

        {/* Heatmap display */}
        <div className="two-cols-step" style={{ marginTop: '16px' }}>
          <div className="panel-box">
            <h4>Heatmap d'Attention Token-à-Token (Aperçu Réel 30×30)</h4>
            <p className="panel-desc">
              {isLoading
                ? 'Extraction des tenseurs d’attention en cours...'
                : attentionData
                ? `Matrice extraite (${attentionData.heads_averaged ? 'Moyenne des têtes' : `Tête #${attentionData.head}`}, Couche ${attentionData.layer})`
                : 'En attente de connexion avec le backend'}
            </p>

            {attentionData?.attention ? (
              <div className="matrix-viewer-box">
                <div
                  className="heatmap-grid-scroll"
                  style={{
                    gridTemplateColumns: `repeat(${Math.min(30, attentionData.attention.length)}, 1fr)`,
                  }}
                >
                  {attentionData.attention.slice(0, 30).flatMap((row, rIdx) =>
                    row.slice(0, 30).map((val, cIdx) => (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        className="matrix-pixel"
                        style={{
                          backgroundColor: `rgba(37, 99, 235, ${Math.min(1, Math.max(0.04, val * 16))})`,
                        }}
                        title={`Query #${rIdx} → Key #${cIdx} : ${val.toFixed(4)}`}
                      />
                    ))
                  )}
                </div>
                <small className="muted-hint" style={{ textAlign: 'center' }}>
                  Survole les pixels pour lire l’affinité de chaque paire de tokens.
                </small>
              </div>
            ) : (
              <div className="empty-attention-cta">
                <Eye size={32} className="text-muted" />
                <b>Matrice en chargement...</b>
              </div>
            )}
          </div>

          <div className="panel-box">
            <h4>Projection Spatiale sur l'Image (Saliency Map du [CLS])</h4>
            <p className="panel-desc">
              Où le token [CLS] concentre-t-il son attention sur les 196 patches pour cette tête et couche ?
            </p>

            {attentionData?.spatial_map ? (
              <div className="spatial-overlay-container">
                <div className="spatial-canvas-relative" style={{ width: '224px', height: '224px' }}>
                  <img src="/presets/samoyed.jpg" alt="Samoyède" className="spatial-base-img" />
                  <div className="spatial-heatmap-grid" style={{ opacity: 0.65 }}>
                    {attentionData.spatial_map.flatMap((row, rIdx) =>
                      row.map((val, cIdx) => (
                        <div
                          key={`${rIdx}-${cIdx}`}
                          className="heatmap-cell"
                          style={{
                            backgroundColor: `rgba(239, 68, 68, ${Math.min(1, Math.max(0.05, val * 1.5))})`,
                          }}
                          title={`Patch (${rIdx}, ${cIdx}) : ${(val * 100).toFixed(1)}%`}
                        />
                      ))
                    )}
                  </div>
                </div>
                <div className="insight-pill" style={{ marginTop: '10px' }}>
                  <span>✔ Les zones rouges indiquent les zones visuelles prépondérantes pour cette tête.</span>
                </div>
              </div>
            ) : (
              <div className="empty-attention-cta">
                <Eye size={32} className="text-muted" />
                <b>Chargement de la projection spatiale...</b>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Navigation footer */}
      <div className="theory-next-bar">
        <span>Étape suivante de compréhension :</span>
        <NavLink to="/comprendre/transformer-encoder" className="next-theory-btn">
          <span>6. L'Encodeur Transformer (Pre-LN & MLP)</span>
          <ArrowRight size={15} />
        </NavLink>
      </div>
    </div>
  );
}
