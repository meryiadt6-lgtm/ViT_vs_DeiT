export const API =
  (import.meta as any).env?.VITE_API_URL || 'http://localhost:8000';

export interface Top5Item {
  class_id: number;
  label: string;
  probability: number;
}

export interface ModelPrediction {
  model_id: string;
  predicted_class_id: number;
  prediction: string;
  confidence: number;
  top5: Top5Item[];
  inference_ms: number;
}

export interface PredictionResult {
  filename: string;
  width: number;
  height: number;
  preview?: string;
  vit: ModelPrediction;
  deit: ModelPrediction;
  agreement: boolean;
}

export interface PredictResponse {
  results: PredictionResult[];
  summary: {
    total_images: number;
    agreements: number;
    disagreements: number;
    agreement_rate: number;
    avg_vit_confidence: number;
    avg_deit_confidence: number;
  };
  device: string;
}

export interface AttentionResponse {
  model: string;
  layer: number;
  head: number;
  heads_count: number;
  tokens: number;
  attention?: number[][];
  spatial_map: number[][];
  dist_spatial_map?: number[][];
  heads_averaged: boolean;
  cls_attention_sample: number[];
}

export interface RolloutResponse {
  model: string;
  rollout_map: number[][];
  tokens: number;
  fusion: string;
}

export interface PreprocessResponse {
  original: {
    width: number;
    height: number;
    preview: string;
  };
  processed: {
    width: number;
    height: number;
    mode: string;
    preview: string;
  };
  channels: {
    red_preview: string;
    green_preview: string;
    blue_preview: string;
    hist_r: number[];
    hist_g: number[];
    hist_b: number[];
    mean_rgb: number[];
    std_rgb: number[];
  };
  tensor: {
    shape: number[];
    min_val: number;
    max_val: number;
    mean_val: number;
    std_val: number;
  };
  patch: {
    index: number;
    row: number;
    col: number;
    size: number;
    pixels_count: number;
    preview: string;
    pixels: number[][][];
    mean_rgb: number[];
  };
}

export interface IntermediateTensorsResponse {
  model: string;
  sample_filters: Array<{
    filter_id: number;
    preview: string;
  }>;
  position_similarity_map: number[][];
  layer_metrics: Array<{
    layer: number;
    norm: number;
    residual_norm: number;
    residual_ratio: number;
  }>;
  deit_dual_head?: {
    cls_head: Array<{ class_id: number; label: string; prob: number }>;
    dist_head: Array<{ class_id: number; label: string; prob: number }>;
    heads_agree: boolean;
  } | null;
}

export interface PerturbationResponse {
  perturb_type: string;
  original_preview: string;
  perturbed_preview: string;
  clean_prediction: ModelPrediction;
  perturbed_prediction: ModelPrediction;
  confidence_drop: number;
  class_changed: boolean;
}

export interface HealthResponse {
  status: string;
  device: string;
  cuda_available: boolean;
  loaded_models: string[];
  models: {
    vit: string;
    deit: string;
  };
}

export async function health(): Promise<HealthResponse> {
  const r = await fetch(`${API}/api/health`);
  if (!r.ok) throw Error('API backend indisponible');
  return r.json();
}

export async function modelInfo() {
  const r = await fetch(`${API}/api/model-info`);
  if (!r.ok) throw Error('Info modèle indisponible');
  return r.json();
}

export async function compare(files: File[]): Promise<PredictResponse> {
  const f = new FormData();
  files.forEach((x) => f.append('files', x));
  const r = await fetch(`${API}/api/predict`, { method: 'POST', body: f });
  const d = await r.json();
  if (!r.ok) throw Error(d.detail || 'Erreur API');
  return d;
}

export async function attention(
  file: File,
  model: string,
  layer: number = 12,
  head: number = 0
): Promise<AttentionResponse> {
  const f = new FormData();
  f.append('file', file);
  const r = await fetch(
    `${API}/api/attention?model_name=${model}&layer=${layer}&head=${head}`,
    { method: 'POST', body: f }
  );
  const d = await r.json();
  if (!r.ok) throw Error(d.detail || 'Extraction de l’attention impossible');
  return d;
}

export async function getAttentionRollout(
  file: File,
  model: string,
  fusion: string = 'mean'
): Promise<RolloutResponse> {
  const f = new FormData();
  f.append('file', file);
  const r = await fetch(
    `${API}/api/attention-rollout?model_name=${model}&head_fusion=${fusion}`,
    { method: 'POST', body: f }
  );
  const d = await r.json();
  if (!r.ok) throw Error(d.detail || 'Calcul du rollout impossible');
  return d;
}

export async function preprocessImage(
  file: File,
  mode: string = 'resize_crop_256',
  patchIdx: number = 86
): Promise<PreprocessResponse> {
  const f = new FormData();
  f.append('file', file);
  const r = await fetch(
    `${API}/api/preprocess?mode=${mode}&patch_idx=${patchIdx}`,
    { method: 'POST', body: f }
  );
  const d = await r.json();
  if (!r.ok) throw Error(d.detail || 'Pré-traitement impossible');
  return d;
}

export async function getIntermediateTensors(
  file: File,
  model: string,
  patchIdx: number = 86
): Promise<IntermediateTensorsResponse> {
  const f = new FormData();
  f.append('file', file);
  const r = await fetch(
    `${API}/api/tensors/intermediate?model_name=${model}&patch_idx=${patchIdx}`,
    { method: 'POST', body: f }
  );
  const d = await r.json();
  if (!r.ok) throw Error(d.detail || 'Extraction des tenseurs impossible');
  return d;
}

export async function runPerturbation(
  file: File,
  perturbType: string = 'mask_center'
): Promise<PerturbationResponse> {
  const f = new FormData();
  f.append('file', file);
  const r = await fetch(
    `${API}/api/perturbation?perturb_type=${perturbType}`,
    { method: 'POST', body: f }
  );
  const d = await r.json();
  if (!r.ok) throw Error(d.detail || 'Échec du test de perturbation');
  return d;
}

/**
 * Helper : Convertit une URL d'image (ex: /presets/bluejay.jpg ou dataUrl) en objet File
 */
export async function urlToFile(url: string, filename: string): Promise<File> {
  const res = await fetch(url);
  const blob = await res.blob();
  const mime = blob.type || 'image/jpeg';
  return new File([blob], filename, { type: mime });
}
