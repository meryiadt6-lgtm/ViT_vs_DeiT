import React from 'react';
import { useLocation, Link } from 'react-router-dom';
import { Home, Sun, Moon, Sparkles } from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export function Header() {
  const location = useLocation();
  const { studentName, backendConnected, theme, toggleTheme, score } = useAppStore();

  const getBreadcrumbTitle = () => {
    const p = location.pathname;
    if (p === '/') return 'Accueil';
    if (p.startsWith('/comprendre')) return 'Comprendre';
    if (p.includes('calcul')) return 'Calculs manuels';
    if (p.startsWith('/pratiquer')) return 'Pratiquer';
    if (p.includes('comparer') || p.includes('compar')) return 'Comparaison';
    if (p.includes('a-propos')) return 'À propos';
    return 'Accueil';
  };

  const initial = (studentName || 'E').trim().charAt(0).toUpperCase();

  return (
    <header className="unified-top-header">
      {/* Breadcrumb Area */}
      <div className="header-breadcrumb">
        <Link to="/" className="breadcrumb-home-link">
          <Home size={16} className="crumb-home-icon" />
          <span>Accueil</span>
        </Link>
        {location.pathname !== '/' && (
          <>
            <span className="crumb-separator">&gt;</span>
            <span className="crumb-active">{getBreadcrumbTitle()}</span>
          </>
        )}
      </div>

      {/* Right Stats & Controls */}
      <div className="header-status-area">
        <button
          type="button"
          className="btn-theme-toggle"
          onClick={toggleTheme}
          title={theme === 'light' ? 'Passer en mode sombre' : 'Passer en mode clair'}
        >
          {theme === 'light' ? <Moon size={16} /> : <Sun size={16} />}
        </button>

        <div className={`status-connection-badge ${backendConnected ? 'connected' : 'disconnected'}`}>
          <span className={`status-dot ${backendConnected ? 'green' : 'amber'}`} />
          <span className="status-badge-text">
            {backendConnected ? 'PyTorch Backend connecté' : 'Backend en attente'}
          </span>
        </div>

        <div className="header-score-pill">
          <Sparkles size={13} className="text-yellow-500" />
          <span>{score} XP</span>
        </div>

        <div className="user-avatar-circle" title={`Profil : ${studentName}`}>
          <span>{initial}</span>
        </div>
      </div>
    </header>
  );
}
