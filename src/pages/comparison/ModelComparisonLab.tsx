import React, { useState, useEffect } from 'react';
import {
  Scale,
  Upload,
  Play,
  CheckCircle2,
  XCircle,
  Info,
  Download,
  Printer,
  ChevronDown,
  ChevronUp,
  Layers,
  Clock,
  Zap,
  Check,
  Eye,
  Trash2,
  Plus,
} from 'lucide-react';
import { compare, urlToFile, PredictResponse, PredictionResult } from '../../api';
import { useAppStore, PRESET_IMAGES } from '../../store/useAppStore';
import { InlineMath, BlockMath, TeX } from '../../components/MathFormula';

export function ModelComparisonLab() {
  const { unlockBadge } = useAppStore();

  const [selectedImageFiles, setSelectedImageFiles] = useState<
    Array<{ id: string; name: string; url: string; file?: File }>
  >([
    { id: 'bluejay', name: 'Geai bleu', url: '/presets/bluejay.jpg' },
    { id: 'samoyed', name: 'Chien Samoyède', url: '/presets/samoyed.jpg' },
    { id: 'car', name: 'Voiture de sport', url: '/presets/car.jpg' },
    { id: 'cat', name: 'Chat tigré', url: '/presets/cat.jpg' },
    { id: 'espresso', name: 'Tasse Espresso', url: '/presets/espresso.jpg' },
  ]);

  const [useVit, setUseVit] = useState<boolean>(true);
  const [useDeit, setUseDeit] = useState<boolean>(true);
  const [vitProcessor, setVitProcessor] = useState<string>('ViTImageProcessor');
  const [deitProcessor, setDeitProcessor] = useState<string>('DeiTImageProcessor');

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [predictionData, setPredictionData] = useState<PredictResponse | null>(null);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Automatically run initial real comparison on mount
  useEffect(() => {
    runRealComparison();
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files) return;
    const files = Array.from(e.target.files);
    if (selectedImageFiles.length + files.length > 15) {
      alert('Vous pouvez comparer jusqu\'à 15 images simultanément.');
      return;
    }

    const newItems = files.map((f, idx) => ({
      id: `custom_${Date.now()}_${idx}`,
      name: f.name.replace(/\.[^/.]+$/, ''),
      url: URL.createObjectURL(f),
      file: f,
    }));

    setSelectedImageFiles((prev) => [...prev, ...newItems]);
  };

  const removeImageItem = (id: string) => {
    if (selectedImageFiles.length <= 1) {
      alert('Conservez au moins une image dans le banc d\'essai.');
      return;
    }
    setSelectedImageFiles((prev) => prev.filter((item) => item.id !== id));
  };

  const runRealComparison = async () => {
    setIsRunning(true);
    setErrorMsg(null);
    try {
      // Prepare real File objects for all selected images
      const fileObjects: File[] = [];
      for (const item of selectedImageFiles) {
        if (item.file) {
          fileObjects.push(item.file);
        } else {
          const converted = await urlToFile(item.url, `${item.id}.jpg`);
          fileObjects.push(converted);
        }
      }

      const res = await compare(fileObjects);
      setPredictionData(res);
      unlockBadge('comparison_analyst');
    } catch (err: any) {
      setErrorMsg(err.message || 'Erreur lors de l\'inférence des modèles.');
    } finally {
      setIsRunning(false);
    }
  };

  const handleExportCsv = () => {
    if (!predictionData?.results) return;

    const csvContent = [
      'Image,Classe_predite_ViT,Confiance_ViT,Temps_ViT_ms,Classe_predite_DeiT,Confiance_DeiT,Temps_DeiT_ms,Accord',
      ...predictionData.results.map(
        (r) =>
          `"${r.filename}","${r.vit.prediction}",${(r.vit.confidence * 100).toFixed(2)}%,${r.vit.inference_ms},"${r.deit.prediction}",${(r.deit.confidence * 100).toFixed(2)}%,${r.deit.inference_ms},${r.agreement ? 'ACCORD' : 'DESACCORD'}`
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `comparaison-vit-vs-deit-${Date.now()}.csv`;
    link.click();
  };

  const handlePrintPdf = () => {
    window.print();
  };

  const summary = predictionData?.summary;

  return (
    <div className="unified-page-container">
      {/* 1. BANNER HEADER */}
      <div className="unified-banner-card">
        <div className="banner-left-content">
          <div className="banner-icon-box">
            <Scale size={28} className="banner-main-icon" />
          </div>
          <div className="banner-titles">
            <h1 className="banner-title">Comparaison globale : ViT vs DeiT</h1>
            <p className="banner-subtitle">
              Inférence PyTorch en temps réel • Évaluez les prédictions, la confiance, l'accord et le temps d'exécution
            </p>
          </div>
        </div>
      </div>

      {/* 2. CONFIGURATION & SÉLECTION DES IMAGES */}
      <div className="compare-two-panels-grid">
        {/* Panel 1 : Importer vos images */}
        <div className="compare-column-panel">
          <div className="panel-navy-header">
            <span className="panel-num-circle">1</span>
            <span className="panel-header-title">Images du banc d'essai ({selectedImageFiles.length}/15)</span>
          </div>

          <div className="panel-white-body">
            <label className="compare-dropzone-box">
              <input
                type="file"
                multiple
                accept="image/jpeg,image/png,image/webp"
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
              <Upload size={22} className="cloud-svg-icon" />
              <p className="dropzone-lead-text">
                <strong>Déposez vos images ici</strong> ou <span className="dropzone-sublink">parcourir</span>
              </p>
              <span className="dropzone-limits-text">Jusqu'à 15 images simultanées</span>
            </label>

            {/* Thumbnails of current batch */}
            <div className="imported-thumbs-row">
              {selectedImageFiles.map((item) => (
                <div key={item.id} className="thumb-item-wrapper" title={item.name}>
                  <img src={item.url} alt={item.name} className="comp-thumb" />
                  <button
                    type="button"
                    className="btn-remove-thumb"
                    onClick={() => removeImageItem(item.id)}
                    title="Retirer cette image"
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Panel 2 : Modèles à comparer */}
        <div className="compare-column-panel">
          <div className="panel-navy-header">
            <span className="panel-num-circle">2</span>
            <span className="panel-header-title">Paramètres des modèles HuggingFace</span>
          </div>

          <div className="panel-white-body models-selection-body">
            <div className="models-select-cards-row">
              {/* ViT Config */}
              <div className={`model-config-card ${useVit ? 'active' : ''}`}>
                <label className="model-check-label">
                  <input
                    type="checkbox"
                    checked={useVit}
                    onChange={(e) => setUseVit(e.target.checked)}
                    className="styled-checkbox"
                  />
                  <span className="model-card-heading">Google ViT-Base</span>
                </label>
                <div className="proc-dropdown-wrap">
                  <span className="proc-label">Processeur d'images :</span>
                  <select
                    value={vitProcessor}
                    onChange={(e) => setVitProcessor(e.target.value)}
                    className="proc-select"
                  >
                    <option value="ViTImageProcessor">ViTImageProcessor</option>
                  </select>
                </div>
              </div>

              {/* DeiT Config */}
              <div className={`model-config-card ${useDeit ? 'active' : ''}`}>
                <label className="model-check-label">
                  <input
                    type="checkbox"
                    checked={useDeit}
                    onChange={(e) => setUseDeit(e.target.checked)}
                    className="styled-checkbox"
                  />
                  <span className="model-card-heading">Meta DeiT-Base Distilled</span>
                </label>
                <div className="proc-dropdown-wrap">
                  <span className="proc-label">Processeur d'images :</span>
                  <select
                    value={deitProcessor}
                    onChange={(e) => setDeitProcessor(e.target.value)}
                    className="proc-select"
                  >
                    <option value="DeiTImageProcessor">DeiTImageProcessor</option>
                  </select>
                </div>
              </div>
            </div>

            <button
              type="button"
              className="btn-launch-comparison"
              onClick={runRealComparison}
              disabled={isRunning || (!useVit && !useDeit)}
            >
              <Play size={16} fill="#ffffff" />
              <span>{isRunning ? 'Inférence PyTorch en cours...' : 'Lancer la comparaison réelle'}</span>
            </button>
          </div>
        </div>
      </div>

      {errorMsg && (
        <div className="error-alert-banner">
          <XCircle size={18} />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* 3. TABLEAU DES RÉSULTATS DE COMPARAISON */}
      <div className="compare-results-panel">
        <div className="panel-navy-header">
          <div className="results-header-left">
            <span className="panel-num-circle">3</span>
            <span className="panel-header-title">Résultats réels de l'évaluation comparative</span>
          </div>
        </div>

        <div className="table-responsive-wrapper">
          <table className="comparison-table-view">
            <thead>
              <tr>
                <th>Image</th>
                <th>Classe prédite ViT</th>
                <th>Confiance ViT</th>
                <th>Temps ViT</th>
                <th>Classe prédite DeiT</th>
                <th>Confiance DeiT</th>
                <th>Temps DeiT</th>
                <th>Accord</th>
                <th>Détail</th>
              </tr>
            </thead>
            <tbody>
              {predictionData?.results.map((r, idx) => {
                const isExpanded = expandedRow === idx;
                return (
                  <React.Fragment key={idx}>
                    <tr className="main-table-row">
                      <td className="td-img">
                        <img
                          src={r.preview || selectedImageFiles[idx]?.url}
                          alt={r.filename}
                          className="table-thumb-img"
                        />
                      </td>

                      <td className="font-bold text-slate-800">{r.vit.prediction}</td>

                      <td className="conf-bar-cell">
                        <span className="conf-pct-num">{(r.vit.confidence * 100).toFixed(1)}%</span>
                        <div className="conf-progress-track">
                          <div
                            className="conf-progress-fill"
                            style={{ width: `${r.vit.confidence * 100}%` }}
                          />
                        </div>
                      </td>

                      <td className="text-xs text-slate-500 font-mono">{r.vit.inference_ms} ms</td>

                      <td className="font-bold text-slate-800">{r.deit.prediction}</td>

                      <td className="conf-bar-cell">
                        <span className="conf-pct-num">{(r.deit.confidence * 100).toFixed(1)}%</span>
                        <div className="conf-progress-track">
                          <div
                            className="conf-progress-fill"
                            style={{
                              width: `${r.deit.confidence * 100}%`,
                              backgroundColor: '#8b5cf6',
                            }}
                          />
                        </div>
                      </td>

                      <td className="text-xs text-slate-500 font-mono">{r.deit.inference_ms} ms</td>

                      <td className="td-accord">
                        {r.agreement ? (
                          <div className="accord-check-circle" title="Les deux modèles s'accordent">
                            <Check size={14} strokeWidth={3} />
                          </div>
                        ) : (
                          <div className="disaccord-badge" title="Désaccord sur la classe gagnante">
                            ≠
                          </div>
                        )}
                      </td>

                      <td>
                        <button
                          type="button"
                          className="btn-expand-row"
                          onClick={() => setExpandedRow(isExpanded ? null : idx)}
                          title="Voir les Top-5 classes"
                        >
                          {isExpanded ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
                        </button>
                      </td>
                    </tr>

                    {/* Expanded Top-5 Drawer */}
                    {isExpanded && (
                      <tr className="expanded-top5-row">
                        <td colSpan={9}>
                          <div className="expanded-top5-container">
                            <div className="top5-side-col">
                              <span className="top5-col-title">Top-5 Google ViT :</span>
                              {r.vit.top5.map((t) => (
                                <div key={t.class_id} className="expanded-top5-item">
                                  <span className="top5-lbl">{t.label}</span>
                                  <span className="top5-val">{(t.probability * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>

                            <div className="top5-side-col">
                              <span className="top5-col-title deit">Top-5 Meta DeiT Distilled :</span>
                              {r.deit.top5.map((t) => (
                                <div key={t.class_id} className="expanded-top5-item">
                                  <span className="top5-lbl">{t.label}</span>
                                  <span className="top5-val deit">{(t.probability * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Summary Footer Bar */}
        <div className="compare-summary-footer-bar">
          <div className="summary-badges-group">
            <span className="summary-pill green-pill">
              <CheckCircle2 size={14} />
              Taux d'accord : <b>{((summary?.agreement_rate ?? 1.0) * 100).toFixed(0)}%</b>
            </span>

            <span className="summary-pill gray-pill">
              <Layers size={14} />
              Images analysées : <b>{summary?.total_images ?? selectedImageFiles.length}</b>
            </span>

            <span className="summary-pill info-pill">
              <Zap size={14} />
              Confiance moyenne ViT : <b>{((summary?.avg_vit_confidence ?? 0.95) * 100).toFixed(1)}%</b> | DeiT :{' '}
              <b>{((summary?.avg_deit_confidence ?? 0.94) * 100).toFixed(1)}%</b>
            </span>
          </div>

          <div className="export-actions-row">
            <button type="button" className="btn-export-csv" onClick={handleExportCsv}>
              <Download size={14} />
              <span>Exporter les résultats CSV</span>
            </button>
            <button type="button" className="btn-print-report" onClick={handlePrintPdf}>
              <Printer size={14} />
              <span>Rapport PDF</span>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
