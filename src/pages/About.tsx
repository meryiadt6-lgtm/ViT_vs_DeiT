import React, { useRef } from 'react';
import {
  Award,
  Crown,
  BookOpen,
  ExternalLink,
  Printer,
  Sparkles,
  CheckCircle2,
  Lock,
  Layers,
  Cpu,
  Brain,
  ShieldCheck,
} from 'lucide-react';
import { useAppStore } from '../store/useAppStore';
import { TeX, InlineMath, BlockMath } from '../components/MathFormula';

export const About: React.FC = () => {
  const { studentName, setStudentName, score, badges, validatedTheorySteps, validatedPracticeSteps } =
    useAppStore();

  const printCertificate = () => {
    window.print();
  };

  const currentDate = new Date().toLocaleDateString('fr-FR', {
    day: 'numeric',
    month: 'long',
    year: 'numeric',
  });

  const theoryCount = Object.keys(validatedTheorySteps).length;
  const practiceCount = Object.keys(validatedPracticeSteps).length;
  const unlockedBadges = badges.filter((b) => b.unlocked);

  return (
    <div className="about-page-container">
      {/* 1. HERO ACADÉMIQUE */}
      <section className="about-hero-card">
        <div className="about-hero-content">
          <span className="hero-kicker-pill">
            <BookOpen size={14} /> FONDATIONS THÉORIQUES &amp; CRÉDITS
          </span>
          <h1>Vision Transformer &amp; DeiT Learning Studio</h1>
          <p>
            Cette plateforme a été conçue comme un laboratoire pédagogique immersif dédié à l'étude des architectures
            Attention-based appliquées à la vision artificielle. Les calculs et les visualisations reposent sur des
            modèles PyTorch réels pré-entraînés sur ImageNet-1k.
          </p>
        </div>
      </section>

      {/* 2. RÉFÉRENCES SCIENTIFIQUES MAJEURES */}
      <section className="academic-papers-section">
        <div className="section-title-wrap">
          <span className="section-kicker">BIBLIOGRAPHIE ACADÉMIQUE</span>
          <h2>Publications fondatrices</h2>
        </div>

        <div className="papers-grid">
          {/* Paper 1: ViT */}
          <div className="paper-card">
            <div className="paper-badge vit-badge">ICLR 2021 • GOOGLE RESEARCH</div>
            <h3>An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale</h3>
            <p className="paper-authors">
              Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner,
              Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby.
            </p>
            <div className="paper-abstract">
              <strong>Résumé clé :</strong> Tandis que l'architecture Transformer est devenue le standard pour le NLP,
              ses applications en vision restaient limitées. Dans ce travail, les auteurs montrent que la dépendance aux
              CNNs n'est pas nécessaire et qu'un Transformer pur appliqué directement à des séquences de patches d'images
              peut surpasser les architectures convolutives de pointe sur ImageNet.
            </div>
            <div className="paper-math-highlight">
              <BlockMath math="\mathbf{z}_0 = [\mathbf{x}_{\text{class}};\, \mathbf{x}_p^1\mathbf{E};\, \dots;\, \mathbf{x}_p^N\mathbf{E}] + \mathbf{E}_{\text{pos}},\quad \mathbf{E} \in \mathbb{R}^{(P^2C) \times D}" />
            </div>
            <a
              href="https://arxiv.org/abs/2010.11929"
              target="_blank"
              rel="noreferrer"
              className="paper-arxiv-link"
            >
              <span>Consulter sur arXiv (2010.11929)</span>
              <ExternalLink size={14} />
            </a>
          </div>

          {/* Paper 2: DeiT */}
          <div className="paper-card">
            <div className="paper-badge deit-badge">ICML 2021 • META AI &amp; SORBONNE</div>
            <h3>Training data-efficient image transformers &amp; distillation through attention</h3>
            <p className="paper-authors">
              Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, Hervé Jégou.
            </p>
            <div className="paper-abstract">
              <strong>Résumé clé :</strong> Les Vision Transformers nécessitent traditionnellement des centaines de
              millions d'images pour converger efficacement. DeiT introduit une procédure d'entraînement basée sur un{' '}
              <strong>token de distillation</strong> qui interagit avec les patchs par auto-attention et apprend à
              reproduire les prédictions d'un réseau enseignant convolutif (RegNetY), entraînable en quelques jours sur
              ImageNet seul.
            </div>
            <div className="paper-math-highlight">
              <BlockMath math="\mathcal{L}_{\text{global}}^{\text{hard}} = \frac{1}{2} \mathcal{L}_{\text{CE}}(\psi(Z_{\text{cls}}), y) + \frac{1}{2} \mathcal{L}_{\text{CE}}(\psi(Z_{\text{dist}}), y_{\text{teacher}})" />
            </div>
            <a
              href="https://arxiv.org/abs/2012.12877"
              target="_blank"
              rel="noreferrer"
              className="paper-arxiv-link"
            >
              <span>Consulter sur arXiv (2012.12877)</span>
              <ExternalLink size={14} />
            </a>
          </div>
        </div>
      </section>

      {/* 3. BIAIS INDUCTIF : CNN VS TRANSFORMERS */}
      <section className="inductive-bias-section">
        <div className="section-title-wrap">
          <span className="section-kicker">ANALYSE ARCHITECTURALE</span>
          <h2>Le Biais Inductif : Pourquoi ViT a-t-il besoin de données massives ou de distillation ?</h2>
        </div>

        <div className="bias-comparative-cards">
          <div className="bias-card">
            <div className="bias-icon cnn-icon">
              <Cpu size={24} />
            </div>
            <h3>Réseaux Convolutifs (CNN)</h3>
            <span className="bias-tag">Fort biais inductif spatial</span>
            <p>
              Les convolutions possèdent deux hypothèses mathématiques fortes gravées dans leur structure :
            </p>
            <ul>
              <li>
                <strong>Localité spatiale :</strong> les pixels voisins sont traités conjointement par des noyaux de petite
                taille ($3\times 3$ ou $5\times 5$).
              </li>
              <li>
                <strong>Équivariance par translation :</strong> le même filtre glisse sur toute l'image, rendant la
                détection d'un motif invariante à sa position.
              </li>
            </ul>
            <div className="bias-note">
              Avantage : Apprennent rapidement sur de petits jeux de données. Inconvénient : Portée réceptive globale
              difficile à propager.
            </div>
          </div>

          <div className="bias-card">
            <div className="bias-icon vit-icon">
              <Brain size={24} />
            </div>
            <h3>Vision Transformers (ViT)</h3>
            <span className="bias-tag">Biais inductif quasi-nul</span>
            <p>
              Les Transformers n'ont aucune connaissance préalable de la structure 2D d'une image :
            </p>
            <ul>
              <li>
                <strong>Champ récepteur global immédiat :</strong> chaque patch peut immédiatement prêter attention à
                n'importe quel autre patch via le mécanisme multi-têtes.
              </li>
              <li>
                <strong>Positions apprises :</strong> l'agencement spatial 2D n'est appris que par les{' '}
                <InlineMath math="\mathbf{E}_{\text{pos}}" />. Si on mélangeait les positions, le modèle apprendrait
                simplement une autre géométrie.
              </li>
            </ul>
            <div className="bias-note">
              Conséquence : ViT requiert JFT-300M pour découvrir la topologie 2D, tandis que <strong>DeiT</strong> contourne ce
              déficit en empruntant le biais inductif d'un CNN enseignant par distillation.
            </div>
          </div>
        </div>
      </section>

      {/* 4. GALERIE DE BADGES DÉBLOQUÉS */}
      <section className="badges-showcase-section">
        <div className="section-title-wrap">
          <span className="section-kicker">TABLEAU DES SUCCÈS</span>
          <h2>Vos Badges et Compétences Débloquées</h2>
        </div>

        <div className="badges-grid-view">
          {badges.map((badge) => {
            return (
              <div
                key={badge.id}
                className={`badge-achievement-card ${badge.unlocked ? 'unlocked' : 'locked'}`}
              >
                <div className="badge-card-icon">
                  {badge.unlocked ? (
                    <Award size={26} className="text-yellow-400" />
                  ) : (
                    <Lock size={22} className="text-slate-400" />
                  )}
                </div>
                <div className="badge-card-content">
                  <div className="badge-card-header">
                    <h4>{badge.name}</h4>
                    {badge.unlocked && <span className="badge-unlocked-tag">Validé</span>}
                  </div>
                  <p>{badge.description}</p>
                  {badge.unlockedAt && (
                    <span className="badge-unlocked-date">
                      Obtenu le {new Date(badge.unlockedAt).toLocaleDateString('fr-FR')}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* 5. CERTIFICAT OFFICIEL IMPRIMABLE */}
      <section className="certificate-section-wrapper">
        <div className="certificate-controls-bar">
          <div className="certificate-student-field">
            <label htmlFor="cert-name-input">Nom sur le certificat :</label>
            <input
              id="cert-name-input"
              type="text"
              value={studentName}
              onChange={(e) => setStudentName(e.target.value)}
              className="cert-name-edit"
              placeholder="Prénom et Nom"
            />
          </div>
          <button type="button" className="btn-print-certificate" onClick={printCertificate}>
            <Printer size={16} />
            <span>Imprimer / Sauvegarder en PDF</span>
          </button>
        </div>

        {/* The Printable Certificate Frame */}
        <div className="official-certificate-frame" id="printable-certificate">
          <div className="certificate-border-inner">
            <div className="cert-header">
              <div className="cert-seal-icon">
                <ShieldCheck size={48} className="text-blue-700" />
              </div>
              <span className="cert-institution">VISION TRANSFORMER &amp; DEIT LEARNING STUDIO</span>
              <h2 className="cert-title">Certificat de Maîtrise en Vision Transformers</h2>
              <span className="cert-subtitle">Approche Théorique &amp; Pratique Avancée</span>
            </div>

            <div className="cert-body">
              <p className="cert-intro">Ce certificat atteste que</p>
              <h3 className="cert-recipient-name">{studentName || 'Étudiant(e) en Intelligence Artificielle'}</h3>
              <p className="cert-statement">
                a complété avec succès les travaux dirigés et pratiques du laboratoire interactif, validant les calculs
                manuels de projection affine, de normalisation de couche, du produit scalaire d'attention multi-têtes
                <InlineMath math="QK^T/\sqrt{d_k}" />, de l'activation GELU et de la distillation d'attention DeiT.
              </p>

              <div className="cert-metrics-row">
                <div className="cert-metric-item">
                  <span className="cert-metric-num">{score} / 100</span>
                  <span className="cert-metric-lbl">Score Global XP</span>
                </div>
                <div className="cert-metric-item">
                  <span className="cert-metric-num">{theoryCount} / 7</span>
                  <span className="cert-metric-lbl">Étapes Théoriques</span>
                </div>
                <div className="cert-metric-item">
                  <span className="cert-metric-num">{practiceCount} / 7</span>
                  <span className="cert-metric-lbl">Étapes Pratiques</span>
                </div>
                <div className="cert-metric-item">
                  <span className="cert-metric-num">{unlockedBadges.length} / {badges.length}</span>
                  <span className="cert-metric-lbl">Badges Validés</span>
                </div>
              </div>
            </div>

            <div className="cert-footer">
              <div className="cert-signature-box">
                <span className="sig-line"></span>
                <span className="sig-role">Direction Académique Deep Learning</span>
                <span className="sig-date">Délivré le {currentDate}</span>
              </div>
              <div className="cert-seal-circle">
                <Crown size={28} className="text-blue-800" />
                <span>OFFICIEL</span>
              </div>
              <div className="cert-signature-box">
                <span className="sig-line"></span>
                <span className="sig-role">Laboratoire Vision Transformer &amp; DeiT</span>
                <span className="sig-date">Code d'authenticité : ViT-DEIT-{Math.abs(score * 1337 + 42)}</span>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};

