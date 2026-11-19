export interface LabStats {
  completedSteps: number[];
  correctAnswersCount: number;
  totalAttemptsCount: number;
  score: number; // 0 to 100
}

export interface ValidationFeedback {
  isChecked: boolean;
  isCorrect: boolean;
  message: string;
  fieldErrors?: Record<string, string>;
}

export type PatchSizeOption = 8 | 16 | 32;
