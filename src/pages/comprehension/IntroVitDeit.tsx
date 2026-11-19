import React, { useState } from 'react';
import {
  BrainCircuit,
  Layers,
  Sparkles,
  ArrowRight,
  Database,
  Eye,
  CheckCircle2,
  Cpu,
  Boxes,
  HelpCircle,
  TrendingUp,
} from 'lucide-react';
import { NavLink } from 'react-router-dom';

export function IntroVitDeit() {
  const [selectedTopic, setSelectedTopic] = useState<'why' | 'cnn_vs_vit' | 'vit_vs_deit'>('why');

  return (
    <div className="theory-page">
      <div className="theory-header">
        <span className="section-pill blue">ESPACE COMPRENDRE · SECTION 01</span>
        <h1>1. Introduction aux Vision Transformers : ViT & DeiT</h1>
        <p className="theory-lead">
          Pendant une décennie, les réseaux convolutifs (CNN) ont dominé la vision par ordinateur.
          En 2020, Google avec <strong>ViT</strong> puis Meta AI en 2021 avec <strong>DeiT</strong> ont prouvé que les architectures Transformers, nées dans le traitement du langage naturel (NLP), pouvaient surpasser les CNN en vision.
        </p>
      </div>

      {/* Navigation tabs for Section 1 */}
      <div className="theory-subtabs">
        <button
          className={`theory-tab-btn ${selectedTopic === 'why' ? 'active' : ''}`}
          onClick={() => setSelectedTopic('why')}
        >
          <HelpCircle size={15} />
          <span>Pourquoi les Transformers pour la vision ?</span>
        </button>
        <button
          className={`theory-tab-btn ${selectedTopic === 'cnn_vs_vit' ? 'active' : ''}`}
          onClick={() => setSelectedTopic('cnn_vs_vit')}
        >
          <Cpu size={15} />
          <span>Différences CNN vs Vision Transformers</span>
        </button>
        <button
          className={`theory-tab-btn ${selectedTopic === 'vit_vs_deit' ? 'active' : ''}`}
          onClick={() => setSelectedTopic('vit_vs_deit')}
        >
          <Sparkles size={15} />
          <span>Comparaison ViT vs DeiT</span>
        </button>
      </div>

      {/* TOPIC 1: Pourquoi utiliser les Transformers pour la vision ? */}
      {selectedTopic === 'why' && (
        <div className="theory-card-stack">
          <div className="comp-card">
            <div className="card-top">
              <div>
                <h3>La Rupture Technologique : De la Convolution au Self-Attention</h3>
                <p>Pourquoi abandonner les filtres convolutifs traditionnels pour des mécanismes d'attention globale ?</p>
              </div>
              <span className="badge-tag">PARADIGME GLOBAL</span>
            </div>

            <div className="two-cols-comp">
              <div className="theory-block">
                <div className="block-icon-header">
                  <Eye className="text-blue" size={22} />
                  <b>Champ Récepteur Global Immédiat</b>
                </div>
                <p>
                  Dans un CNN, un neurone de la première couche ne voit qu'une infime fenêtre (ex. 3×3 pixels).
                  Il faut empiler des dizaines de couches pour que le champ récepteur s'étende à toute l'image.
                </p>
                <div className="insight-pill">
                  <span>✔ Avec ViT, chaque patch de l'image peut communiquer directement avec n'importe quel autre patch dès la première couche de Self-Attention.</span>
                </div>
              </div>

              <div className="theory-block">
                <div className="block-icon-header">
                  <TrendingUp className="text-purple" size={22} />
                  <b>Capacité de Passage à l'Échelle (Scaling Laws)</b>
                </div>
                <p>
                  Les convolutions souffrent d'une saturation de performance lorsque la quantité de données d'entraînement augmente au-delà de 100 millions d'images.
                </p>
                <div className="insight-pill purple">
                  <span>✔ Les Vision Transformers continuent d'améliorer leur précision à mesure que les données et le calcul augmentent, sans plafonner.</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* TOPIC 2: Différences CNN vs Vision Transformers */}
      {selectedTopic === 'cnn_vs_vit' && (
        <div className="theory-card-stack">
          <div className="comp-card">
            <div className="card-top">
              <div>
                <h3>Biais Inductif : Le Cœur de la Différence CNN vs ViT</h3>
                <p>Comparatif des propriétés fondamentales d'apprentissage.</p>
              </div>
              <span className="badge-outline">ANALYSE FONDAMENTALE</span>
            </div>

            <div className="comparison-table-wrap">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Caractéristique</th>
                    <th>Convolutional Neural Networks (CNN)</th>
                    <th>Vision Transformers (ViT)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td><b>Biais Inductif (Inductive Bias)</b></td>
                    <td><span className="pill-strong">Très fort</span> : Localité spatiale et invariance par translation intégrées par conception.</td>
                    <td><span className="pill-weak">Quasi inexistant</span> : Le modèle n'a aucune hypothèse préalable sur la structure 2D des images.</td>
                  </tr>
                  <tr>
                    <td><b>Traitement de l'Image</b></td>
                    <td>Filtres glissants de convolutions locales (3×3, 5×5).</td>
                    <td>Découpage en grille de patches indépendants (16×16 px) traités comme une séquence de tokens.</td>
                  </tr>
                  <tr>
                    <td><b>Portée des Relations</b></td>
                    <td>Locale d'abord, globale seulement dans les couches très profondes.</td>
                    <td>Globale dès la couche 1 via le mécanisme de Self-Attention multi-têtes.</td>
                  </tr>
                  <tr>
                    <td><b>Besoin en Données</b></td>
                    <td>Fonctionne bien sur des datasets de taille moyenne (ImageNet-1k, 1.2M images).</td>
                    <td>Sur-apprend rapidement sur petits datasets ; nécessite de très grands volumes de données pré-entraînées sans distillation.</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TOPIC 3: Comparaison entre ViT et DeiT */}
      {selectedTopic === 'vit_vs_deit' && (
        <div className="theory-card-stack">
          <div className="comp-card">
            <div className="card-top">
              <div>
                <h3>ViT (Google) vs DeiT (Meta / Facebook) : La Percée de la Distillation</h3>
                <p>Comment DeiT a rendu les Vision Transformers accessibles sans supercalculateur géant.</p>
              </div>
              <span className="badge-tag deit-tag">DATA-EFFICIENT</span>
            </div>

            <div className="two-cols-comp">
              <div className="model-spotlight vit-spotlight">
                <div className="spotlight-badge vit-badge">GOOGLE RESEARCH (2020)</div>
                <h4>Google ViT (Vision Transformer)</h4>
                <p>
                  Dosovitskiy et al. ont proposé l'architecture fondatrice. Cependant, pour battre les ResNet, ViT a dû être pré-entraîné sur <strong>JFT-300M (300 millions d'images propriétaires Google)</strong>.
                </p>
                <div className="spotlight-features">
                  <div><span>Tokens d'entrée :</span> <b>197 (1 CLS + 196 Patches)</b></div>
                  <div><span>Données requises :</span> <b>JFT-300M ou ImageNet-21k</b></div>
                  <div><span>Tête de décision :</span> <b>Tête MLP unique sur le token CLS</b></div>
                </div>
              </div>

              <div className="model-spotlight deit-spotlight">
                <div className="spotlight-badge deit-badge">META AI / SORBONNE (2021)</div>
                <h4>Meta DeiT (Data-efficient Image Transformer)</h4>
                <p>
                  Touvron et al. ont résolu le problème de gourmandise en données grâce à la <strong>distillation de connaissances</strong> guidée par un ConvNet (Teacher), permettant d'atteindre 85.2% sur <strong>ImageNet-1k seul</strong>.
                </p>
                <div className="spotlight-features">
                  <div><span>Tokens d'entrée :</span> <b>198 (1 CLS + 1 DIST + 196 Patches)</b></div>
                  <div><span>Données requises :</span> <b>ImageNet-1k standard (1.2M images)</b></div>
                  <div><span>Tête de décision :</span> <b>Double tête (CLS + DIST) moyennée</b></div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Quick Navigation Footer to next section */}
      <div className="theory-next-bar">
        <span>Étape suivante de compréhension :</span>
        <NavLink to="/comprendre/architecture" className="next-theory-btn">
          <span>2. Architecture Globale du Pipeline</span>
          <ArrowRight size={15} />
        </NavLink>
      </div>
    </div>
  );
}
