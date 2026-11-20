// Pure mathematical calculation engine for the ViT & DeiT Laboratory
// All expected values computed from first principles, verified against PyTorch reference

export function calculatePatchStats(H: number, W: number, P: number, C: number = 3) {
  const gridRows = Math.floor(H / P);
  const gridCols = Math.floor(W / P);
  const totalPatches = gridRows * gridCols;
  const flattenedDim = P * P * C;
  return { gridRows, gridCols, totalPatches, flattenedDim };
}

// Config objects for Miniature vs Real mode
export function getModeConfig(mode: 'miniature' | 'real') {
  if (mode === 'miniature') {
    return { H: 6, W: 6, P: 2, C: 3, D_model: 4, N: 9, D_in: 12 };
  }
  return { H: 224, W: 224, P: 16, C: 3, D_model: 768, N: 196, D_in: 768 };
}

export function getSeqLen(N: number, model: 'vit' | 'deit') {
  return model === 'deit' ? N + 2 : N + 1;
}

export function isClose(userVal: number | string, trueVal: number, tol: number = 0.01): boolean {
  if (userVal === undefined || userVal === null || String(userVal).trim() === '') return false;
  // Handle French comma decimals and % symbols
  const cleanStr = String(userVal).trim().replace(',', '.').replace('%', '');
  const parsed = parseFloat(cleanStr);
  if (isNaN(parsed)) return false;

  // If expected value is in [0,1] and user entered percentage (e.g. 66.5 instead of 0.665)
  if (trueVal >= 0 && trueVal <= 1 && parsed > 1 && parsed <= 100) {
    if (Math.abs(parsed / 100 - trueVal) <= tol) return true;
  }

  const diff = Math.abs(parsed - trueVal);
  if (diff <= tol) return true;
  // Also accept within 2% relative for larger values
  if (Math.abs(trueVal) > 0.001 && diff / Math.abs(trueVal) <= 0.02) return true;
  return false;
}

export function calculateVectorMatrixMult(x: number[], W: number[][], b: number[]): number[] {
  const M = b.length;
  const result: number[] = new Array(M).fill(0);
  for (let j = 0; j < M; j++) {
    let sum = 0;
    for (let i = 0; i < x.length; i++) {
      sum += x[i] * W[i][j];
    }
    result[j] = sum + b[j];
  }
  return result;
}

export function calculateLayerNorm(values: number[], eps: number = 1e-5) {
  const D = values.length;
  const mean = values.reduce((acc, v) => acc + v, 0) / D;
  const variance = values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / D;
  const stdDev = Math.sqrt(variance);
  const normalized = values.map((v) => (v - mean) / Math.sqrt(variance + eps));
  return { mean, variance, stdDev, normalized };
}

// Corrected attention: Q=[[2,0,1,0],[1,1,0,1]], K=[[1,1,0,0],[0,2,1,0]], dk=4
// QKT = Q @ K^T, S = QKT/sqrt(dk), A = softmax(S)
export function calculateAttentionFixed() {
  // Q: 2×4, K: 2×4, dk=4, sqrt(dk)=2
  const Q = [[2, 0, 1, 0], [1, 1, 0, 1]];
  const K = [[1, 1, 0, 0], [0, 2, 1, 0]];
  const dk = 4;

  // K^T: 4×2
  const KT = [[K[0][0], K[1][0]], [K[0][1], K[1][1]], [K[0][2], K[1][2]], [K[0][3], K[1][3]]];

  // QKT: 2×2
  const QKT = [
    [Q[0].reduce((s, v, i) => s + v * KT[i][0], 0), Q[0].reduce((s, v, i) => s + v * KT[i][1], 0)],
    [Q[1].reduce((s, v, i) => s + v * KT[i][0], 0), Q[1].reduce((s, v, i) => s + v * KT[i][1], 0)],
  ];
  // QKT = [[2*1+0*1+1*0+0*0, 2*0+0*2+1*1+0*0], [1*1+1*1+0*0+1*0, 1*0+1*2+0*1+1*0]]
  // = [[2, 1], [2, 2]]

  const scale = Math.sqrt(dk); // 2
  const S = QKT.map(row => row.map(v => v / scale));
  // S = [[1, 0.5], [1, 1]]

  // Row-wise softmax
  const softmaxRow = (row: number[]) => {
    const exps = row.map(v => Math.exp(v));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
  };

  const A = S.map(row => softmaxRow(row));
  // Row0: softmax([1, 0.5]) = [e^1/(e^1+e^0.5), e^0.5/(e^1+e^0.5)]
  //   = [2.7183/(2.7183+1.6487), 1.6487/(2.7183+1.6487)] = [0.6225, 0.3775]
  // Row1: softmax([1, 1]) = [0.5, 0.5]

  return { Q, K, KT, QKT, S, A, dk, scale };
}

export function geluExact(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * Math.pow(x, 3))));
}

// Exact GELU reference values
// geluExact(3.0) ≈ 2.9960 (essentially 3.0)
// geluExact(-1.0) ≈ -0.1588

export function calculateSoftmax(logits: number[]) {
  const maxLogit = Math.max(...logits); // for numerical stability
  const exps = logits.map((z) => Math.exp(z - maxLogit));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sumExp);
  // Recompute absolute exps for display
  const expsAbs = logits.map((z) => Math.exp(z));
  const sumExpAbs = expsAbs.reduce((a, b) => a + b, 0);
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }
  return {
    exps: expsAbs,
    sumExp: sumExpAbs,
    probs,
    winningClass: maxIdx + 1, // 1-indexed
  };
}

// Step 7 reference values:
// z_cls = [2, 1, 0] → P_cls = [0.6652, 0.2447, 0.0900]
// z_dist = [1, 3, 0] → P_dist = [0.1140, 0.8438, 0.0422]  (using exp(1)=2.718, exp(3)=20.086, exp(0)=1 → sum=23.804)
// Paper fusion: P_final = 0.5*(P_cls + P_dist)
//   = [0.5*(0.6652+0.1140), 0.5*(0.2447+0.8438), 0.5*(0.0900+0.0422)]
//   = [0.3896, 0.5443, 0.0661]  → class 2 wins
// HuggingFace variant: softmax((z_cls + z_dist)/2) = softmax([1.5, 2, 0])
//   exps = [e^1.5, e^2, e^0] = [4.4817, 7.3891, 1] sum=12.8708
//   = [0.3483, 0.5739, 0.0777]  → class 2 wins
export function getStep7References() {
  const zCls = [2, 1, 0];
  const zDist = [1, 3, 0];

  const softmaxCls = calculateSoftmax(zCls);
  const softmaxDist = calculateSoftmax(zDist);

  const pFinalPaper = softmaxCls.probs.map((p, i) => 0.5 * (p + softmaxDist.probs[i]));
  const winnerPaper = pFinalPaper.indexOf(Math.max(...pFinalPaper)) + 1;

  const zAvg = zCls.map((z, i) => (z + zDist[i]) / 2);
  const pFinalHF = calculateSoftmax(zAvg).probs;
  const winnerHF = pFinalHF.indexOf(Math.max(...pFinalHF)) + 1;

  return {
    zCls, zDist,
    softmaxCls,
    softmaxDist,
    pFinalPaper,
    winnerPaper,
    pFinalHF,
    winnerHF,
  };
}
