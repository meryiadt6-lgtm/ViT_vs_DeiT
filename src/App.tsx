import React, { useEffect } from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Header } from './components/Header';
import { health } from './api';
import { useAppStore } from './store/useAppStore';

// Pages
import { Home } from './pages/Home';
import { ArchitectureDiagram } from './pages/comprehension/ArchitectureDiagram';
import { TheoreticalCalculations } from './pages/comprehension/TheoreticalCalculations';
import { PracticeWorkflow } from './pages/practice/PracticeWorkflow';
import { ModelComparisonLab } from './pages/comparison/ModelComparisonLab';
import { About } from './pages/About';

// Individual deep-dive theoretical sections
import { IntroVitDeit } from './pages/comprehension/IntroVitDeit';
import { GlobalArchitecture } from './pages/comprehension/GlobalArchitecture';
import { PatchEmbeddingTheory } from './pages/comprehension/PatchEmbeddingTheory';
import { PositionalTokensTheory } from './pages/comprehension/PositionalTokensTheory';
import { SelfAttentionTheory } from './pages/comprehension/SelfAttentionTheory';
import { EncoderTheory } from './pages/comprehension/EncoderTheory';
import { ClassificationTheory } from './pages/comprehension/ClassificationTheory';

export default function App() {
  const { setBackendConnected } = useAppStore();

  useEffect(() => {
    async function checkHealth() {
      try {
        const res = await health();
        if (res.status === 'ok') {
          setBackendConnected(true);
        }
      } catch {
        setBackendConnected(false);
      }
    }
    checkHealth();
    const interval = setInterval(checkHealth, 30000);
    return () => clearInterval(interval);
  }, [setBackendConnected]);

  return (
    <div className="app-container">
      <Sidebar />
      <div className="main-content-layout">
        <Header />
        <main className="page-body">
          <Routes>
            {/* 1. ACCUEIL */}
            <Route path="/" element={<Home />} />
            <Route path="/accueil" element={<Navigate to="/" replace />} />

            {/* 2. COMPRENDRE */}
            <Route path="/comprendre" element={<ArchitectureDiagram />} />
            <Route path="/comprendre/architecture-diagram" element={<ArchitectureDiagram />} />
            <Route path="/comprendre/intro" element={<IntroVitDeit />} />
            <Route path="/comprendre/architecture" element={<GlobalArchitecture />} />
            <Route path="/comprendre/patch-embedding" element={<PatchEmbeddingTheory />} />
            <Route path="/comprendre/position-tokens" element={<PositionalTokensTheory />} />
            <Route path="/comprendre/self-attention" element={<SelfAttentionTheory />} />
            <Route path="/comprendre/transformer-encoder" element={<EncoderTheory />} />
            <Route path="/comprendre/classification" element={<ClassificationTheory />} />

            {/* 3. CALCULS MANUELS (LABORATOIRE THÉORIQUE) */}
            <Route path="/calculs-manuels" element={<TheoreticalCalculations />} />
            <Route path="/comprendre/calculs-theoriques" element={<TheoreticalCalculations />} />

            {/* 4. PRATIQUER (LABORATOIRE PRATIQUE SUR VRAI MODÈLE) */}
            <Route path="/pratiquer" element={<PracticeWorkflow />} />

            {/* 5. COMPARAISON (VIT VS DEIT) */}
            <Route path="/comparer" element={<ModelComparisonLab />} />
            <Route path="/comparaison" element={<Navigate to="/comparer" replace />} />

            {/* 6. À PROPOS */}
            <Route path="/a-propos" element={<About />} />

            {/* Fallbacks */}
            <Route path="/image" element={<Navigate to="/pratiquer" replace />} />
            <Route path="/patches" element={<Navigate to="/comprendre/patch-embedding" replace />} />
            <Route path="/architecture" element={<Navigate to="/comprendre/architecture" replace />} />
            <Route path="/attention" element={<Navigate to="/comprendre/self-attention" replace />} />
            <Route path="/distillation" element={<Navigate to="/comprendre/architecture" replace />} />
            <Route path="/compare" element={<Navigate to="/comparer" replace />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>

        <footer className="global-footer">
          <div className="footer-left">
            <b>VISION TRANSFORMER &amp; DEIT LEARNING STUDIO</b>
            <span>Plateforme pédagogique interactive · Google ViT-Base &amp; Meta DeiT-Base Distilled</span>
          </div>
          <div className="footer-right">
            <span>Projet académique Deep Learning · Attention Multi-Têtes &amp; Distillation Visuelle</span>
          </div>
        </footer>
      </div>
    </div>
  );
}
