// Pure mathematical calculation engine for the ViT & DeiT Laboratory

export function calculatePatchStats(H: number, W: number, P: number, C: number = 3) {
  const gridRows = Math.floor(H / P);
  const gridCols = Math.floor(W / P);
  const totalPatches = gridRows * gridCols;
  const flattenedDim = P * P * C;
  return {
    gridRows,
    gridCols,
    totalPatches,
    flattenedDim,
  };
}

export function isClose(userVal: number | string, trueVal: number, tol: number = 0.05): boolean {
  if (userVal === undefined || userVal === null || String(userVal).trim() === '') return false;
  // Gérer la virgule française et les symboles %
  const cleanStr = String(userVal).trim().replace(',', '.').replace('%', '');
  const parsed = parseFloat(cleanStr);
  if (isNaN(parsed)) return false;

  // Si la valeur attendue est entre 0 et 1 (ex: probabilité), et que l'utilisateur a entré en % (ex: 66.5 au lieu de 0.665)
  if (trueVal >= 0 && trueVal <= 1 && parsed > 1 && parsed <= 100) {
    if (Math.abs(parsed / 100 - trueVal) <= tol) return true;
  }

  const diff = Math.abs(parsed - trueVal);
  if (diff <= tol) return true;
  if (Math.abs(trueVal) > 0.0001 && diff / Math.abs(trueVal) <= 0.08) return true;
  return false;
}

export function calculateVectorMatrixMult(x: number[], W: number[][], b: number[]): number[] {
  // x is [1 x N], W is [N x M], b is [M]
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

export function calculateLayerNorm(values: number[]) {
  const D = values.length;
  const mean = values.reduce((acc, v) => acc + v, 0) / D;
  const variance = values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / D;
  const stdDev = Math.sqrt(variance);
  const normalized = values.map((v) => (v - mean) / Math.sqrt(variance + 1e-5));
  return {
    mean,
    variance,
    stdDev,
    normalized,
  };
}

export function calculateAttention2x2(
  Q: number[][],
  K: number[][],
  V: number[][],
  dk: number
) {
  // K transpose
  const KT = [
    [K[0][0], K[1][0]],
    [K[0][1], K[1][1]],
  ];

  // Q * KT
  const QKT = [
    [Q[0][0] * KT[0][0] + Q[0][1] * KT[1][0], Q[0][0] * KT[0][1] + Q[0][1] * KT[1][1]],
    [Q[1][0] * KT[0][0] + Q[1][1] * KT[1][0], Q[1][0] * KT[0][1] + Q[1][1] * KT[1][1]],
  ];

  const scale = Math.sqrt(dk);
  const scaled = [
    [QKT[0][0] / scale, QKT[0][1] / scale],
    [QKT[1][0] / scale, QKT[1][1] / scale],
  ];

  // Row-wise Softmax
  const expRow0 = [Math.exp(scaled[0][0]), Math.exp(scaled[0][1])];
  const sumExp0 = expRow0[0] + expRow0[1];
  const softmaxRow0 = [expRow0[0] / sumExp0, expRow0[1] / sumExp0];

  const expRow1 = [Math.exp(scaled[1][0]), Math.exp(scaled[1][1])];
  const sumExp1 = expRow1[0] + expRow1[1];
  const softmaxRow1 = [expRow1[0] / sumExp1, expRow1[1] / sumExp1];

  const A = [softmaxRow0, softmaxRow1];

  // Output = A * V
  const out = [
    [
      A[0][0] * V[0][0] + A[0][1] * V[1][0],
      A[0][0] * V[0][1] + A[0][1] * V[1][1],
    ],
    [
      A[1][0] * V[0][0] + A[1][1] * V[1][0],
      A[1][0] * V[0][1] + A[1][1] * V[1][1],
    ],
  ];

  return {
    KT,
    QKT,
    scaled,
    A,
    out,
  };
}

export function geluExact(x: number): number {
  const c = Math.sqrt(2 / Math.PI);
  return 0.5 * x * (1 + Math.tanh(c * (x + 0.044715 * Math.pow(x, 3))));
}

export function calculateSoftmax(logits: number[]) {
  const exps = logits.map((z) => Math.exp(z));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  const probs = exps.map((e) => e / sumExp);
  let maxIdx = 0;
  for (let i = 1; i < probs.length; i++) {
    if (probs[i] > probs[maxIdx]) maxIdx = i;
  }
  return {
    exps,
    sumExp,
    probs,
    winningClass: maxIdx + 1,
  };
}
