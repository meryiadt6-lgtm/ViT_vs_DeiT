import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  Home,
  BookOpen,
  Calculator,
  Play,
  BarChart3,
  Info,
  GraduationCap,
  Award,
} from 'lucide-react';
import { useAppStore } from '../store/useAppStore';

export function Sidebar() {
  const location = useLocation();
  const { score, badges } = useAppStore();
  const unlockedCount = badges.filter((b) => b.unlocked).length;

  return (
    <aside className="studio-sidebar">
      {/* Brand Header */}
      <div className="sidebar-brand">
        <div className="brand-logo-container">
          <GraduationCap size={28} className="brand-cap-icon" />
        </div>
        <div className="brand-titles">
          <span className="brand-name">Vision Transformer</span>
          <span className="brand-model-title">&amp; DeiT</span>
          <span className="brand-subname">Learning Studio</span>
        </div>
      </div>

      {/* Primary Navigation */}
      <nav className="sidebar-nav">
        <NavLink
          to="/"
          end
          className={({ isActive }) =>
            `nav-pill-item ${isActive ? 'active-pill' : ''}`
          }
        >
          <div className="nav-pill-icon">
            <Home size={18} />
          </div>
          <div className="nav-pill-text">
            <div className="nav-pill-label">Accueil</div>
          </div>
        </NavLink>

        <NavLink
          to="/comprendre"
          className={({ isActive }) =>
            `nav-pill-item ${isActive || location.pathname.startsWith('/comprendre') ? 'active-pill' : ''}`
          }
        >
          <div className="nav-pill-icon">
            <BookOpen size={18} />
          </div>
          <div className="nav-pill-text">
            <div className="nav-pill-label">Comprendre</div>
          </div>
        </NavLink>

        <NavLink
          to="/calculs-manuels"
          className={({ isActive }) =>
            `nav-pill-item ${isActive || location.pathname.includes('calcul') ? 'active-pill' : ''}`
          }
        >
          <div className="nav-pill-icon">
            <Calculator size={18} />
          </div>
          <div className="nav-pill-text">
            <div className="nav-pill-label">Calculs manuels</div>
          </div>
        </NavLink>

        <NavLink
          to="/pratiquer"
          className={({ isActive }) =>
            `nav-pill-item ${isActive || location.pathname.startsWith('/pratiquer') ? 'active-pill' : ''}`
          }
        >
          <div className="nav-pill-icon">
            <Play size={18} />
          </div>
          <div className="nav-pill-text">
            <div className="nav-pill-label">Pratiquer</div>
          </div>
        </NavLink>

        <NavLink
          to="/comparer"
          className={({ isActive }) =>
            `nav-pill-item ${isActive || location.pathname.includes('compar') ? 'active-pill' : ''}`
          }
        >
          <div className="nav-pill-icon">
            <BarChart3 size={18} />
          </div>
          <div className="nav-pill-text">
            <div className="nav-pill-label">Comparaison</div>
          </div>
        </NavLink>

        <NavLink
          to="/a-propos"
          className={({ isActive }) =>
            `nav-pill-item ${isActive ? 'active-pill' : ''}`
          }
        >
          <div className="nav-pill-icon">
            <Info size={18} />
          </div>
          <div className="nav-pill-text">
            <div className="nav-pill-label">À propos</div>
          </div>
        </NavLink>
      </nav>

      {/* Gamification Quick Widget in Sidebar */}
      <div className="sidebar-gamification-card">
        <div className="gamify-header">
          <Award size={14} className="text-yellow-400" />
          <span>Score Apprenant</span>
        </div>
        <div className="gamify-score-val">{score} / 100 XP</div>
        <div className="gamify-badges-val">{unlockedCount} / {badges.length} badges validés</div>
      </div>

      {/* Bottom Neural Net Graphic & Tagline */}
      <div className="sidebar-bottom-widget">
        <div className="neural-net-graphic">
          <svg viewBox="0 0 80 80" className="neural-svg" fill="none">
            <line x1="20" y1="20" x2="60" y2="20" stroke="#3b82f6" strokeWidth="1" strokeDasharray="2 2" opacity="0.6" />
            <line x1="20" y1="20" x2="40" y2="40" stroke="#60a5fa" strokeWidth="1.2" opacity="0.7" />
            <line x1="60" y1="20" x2="40" y2="40" stroke="#60a5fa" strokeWidth="1.2" opacity="0.7" />
            <line x1="20" y1="60" x2="40" y2="40" stroke="#60a5fa" strokeWidth="1.2" opacity="0.7" />
            <line x1="60" y1="60" x2="40" y2="40" stroke="#60a5fa" strokeWidth="1.2" opacity="0.7" />
            <line x1="20" y1="60" x2="60" y2="60" stroke="#3b82f6" strokeWidth="1" strokeDasharray="2 2" opacity="0.6" />
            <line x1="20" y1="20" x2="20" y2="60" stroke="#3b82f6" strokeWidth="1" opacity="0.5" />
            <line x1="60" y1="20" x2="60" y2="60" stroke="#3b82f6" strokeWidth="1" opacity="0.5" />
            
            <circle cx="20" cy="20" r="4.5" fill="#38bdf8" />
            <circle cx="60" cy="20" r="4.5" fill="#38bdf8" />
            <circle cx="40" cy="40" r="6" fill="#60a5fa" />
            <circle cx="20" cy="60" r="4.5" fill="#38bdf8" />
            <circle cx="60" cy="60" r="4.5" fill="#38bdf8" />
          </svg>
        </div>
        <span className="sidebar-tagline">Apprendre • Explorer • Maîtriser</span>
      </div>
    </aside>
  );
}
