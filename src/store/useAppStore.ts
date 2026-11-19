import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface PresetImage {
  id: string;
  name: string;
  category: string;
  url: string;
  filename: string;
  trueClass: string;
}

export const PRESET_IMAGES: PresetImage[] = [
  {
    id: 'bluejay',
    name: 'Geai bleu',
    category: 'Oiseaux',
    url: '/presets/bluejay.jpg',
    filename: 'bluejay.jpg',
    trueClass: 'jay (geai bleu)',
  },
  {
    id: 'kingfisher',
    name: 'Martin-pêcheur',
    category: 'Oiseaux',
    url: '/presets/kingfisher.jpg',
    filename: 'kingfisher.jpg',
    trueClass: 'kingfisher (martin-pêcheur)',
  },
  {
    id: 'cat',
    name: 'Chat domestique',
    category: 'Animaux',
    url: '/presets/cat.jpg',
    filename: 'cat.jpg',
    trueClass: 'tabby cat (chat tigré)',
  },
  {
    id: 'samoyed',
    name: 'Chien Samoyède',
    category: 'Animaux',
    url: '/presets/samoyed.jpg',
    filename: 'samoyed.jpg',
    trueClass: 'Samoyed (samoyède)',
  },
  {
    id: 'car',
    name: 'Voiture bleue',
    category: 'Véhicules',
    url: '/presets/car.jpg',
    filename: 'car.jpg',
    trueClass: 'sports car (voiture de sport)',
  },
  {
    id: 'espresso',
    name: 'Tasse Espresso',
    category: 'Objets',
    url: '/presets/espresso.jpg',
    filename: 'espresso.jpg',
    trueClass: 'espresso (café)',
  },
  {
    id: 'shark',
    name: 'Requin blanc',
    category: 'Animaux',
    url: '/presets/shark.jpg',
    filename: 'shark.jpg',
    trueClass: 'great white shark (requin blanc)',
  },
];

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: 'theory' | 'practice' | 'comparison' | 'master';
  unlocked: boolean;
  unlockedAt?: string;
}

export const INITIAL_BADGES: Badge[] = [
  {
    id: 'patch_pioneer',
    name: 'Pionnier des Patches',
    description: 'A découpé une image en 196 patches et calculé leur dimension aplatie.',
    icon: 'Grid',
    category: 'theory',
    unlocked: false,
  },
  {
    id: 'linear_projectionist',
    name: 'Maître de la Projection',
    description: 'A réalisé une projection linéaire affine vers la dimension latente D=768.',
    icon: 'Maximize2',
    category: 'theory',
    unlocked: false,
  },
  {
    id: 'token_architect',
    name: 'Architecte des Tokens',
    description: 'A assemblé la séquence complète avec le token [CLS] et [DIST] de DeiT.',
    icon: 'Layers',
    category: 'theory',
    unlocked: false,
  },
  {
    id: 'norm_master',
    name: 'Expert LayerNorm',
    description: 'A calculé la moyenne, la variance et la normalisation d\'un vecteur latent.',
    icon: 'BarChart2',
    category: 'theory',
    unlocked: false,
  },
  {
    id: 'attention_seeker',
    name: 'Chasseur d\'Attention',
    description: 'A calculé le produit scalaire QKᵀ mis à l\'échelle et le Softmax.',
    icon: 'Zap',
    category: 'theory',
    unlocked: false,
  },
  {
    id: 'gelu_activator',
    name: 'Alchimiste GELU',
    description: 'A manipulé l\'activation non linéaire GELU et l\'expansion MLP 3072D.',
    icon: 'Activity',
    category: 'theory',
    unlocked: false,
  },
  {
    id: 'classifier_champion',
    name: 'Oracle de Classification',
    description: 'A calculé les probabilités Softmax finales et la distillation DeiT.',
    icon: 'Award',
    category: 'theory',
    unlocked: false,
  },
  {
    id: 'practice_explorer',
    name: 'Praticien PyTorch',
    description: 'A exécuté un pipeline de vision complet sur un véritable modèle Transformers.',
    icon: 'Cpu',
    category: 'practice',
    unlocked: false,
  },
  {
    id: 'comparison_analyst',
    name: 'Comparateur ViT vs DeiT',
    description: 'A analysé un lot d\'images et évalué le taux d\'accord entre les deux modèles.',
    icon: 'Scale',
    category: 'comparison',
    unlocked: false,
  },
  {
    id: 'grand_master',
    name: 'Grand Maître Vision Transformer',
    description: 'A validé l\'intégralité du studio théorique et pratique avec un score de 100%.',
    icon: 'Crown',
    category: 'master',
    unlocked: false,
  },
];

export interface Top5Item {
  class_id: number;
  label: string;
  probability: number;
}

export interface ModelInferenceResult {
  model: string;
  model_id: string;
  predicted_class_id: number;
  prediction: string;
  confidence: number;
  top5: Top5Item[];
  inference_ms: number;
}

interface AppState {
  // Image Active
  selectedPreset: string;
  imageUrl: string;
  imageFilename: string;
  customImageData: string | null;
  preprocessMode: 'direct_224' | 'resize_crop_256';
  patchSize: number; // 8, 16, 32

  // Model & Difficulty Mode
  selectedModel: 'vit' | 'deit';
  diffLevel: 'miniature' | 'real';

  // Inferences
  vitResult: ModelInferenceResult | null;
  deitResult: ModelInferenceResult | null;
  isInferring: boolean;
  backendConnected: boolean;

  // Gamification & Progression
  score: number;
  attempts: number;
  validatedTheorySteps: Record<number, boolean>;
  validatedPracticeSteps: Record<number, boolean>;
  badges: Badge[];
  studentName: string;

  // UI
  theme: 'light' | 'dark';
  language: 'fr' | 'en';

  // Actions
  selectPreset: (presetId: string) => void;
  setCustomImage: (dataUrl: string, filename: string) => void;
  setPreprocessMode: (mode: 'direct_224' | 'resize_crop_256') => void;
  setPatchSize: (size: number) => void;
  setSelectedModel: (model: 'vit' | 'deit') => void;
  setDiffLevel: (level: 'miniature' | 'real') => void;
  setInferenceResults: (vit: ModelInferenceResult | null, deit: ModelInferenceResult | null) => void;
  setIsInferring: (status: boolean) => void;
  setBackendConnected: (status: boolean) => void;
  validateTheoryStep: (stepId: number, points?: number) => void;
  validatePracticeStep: (stepId: number) => void;
  registerAttempt: () => void;
  unlockBadge: (badgeId: string) => void;
  setStudentName: (name: string) => void;
  toggleTheme: () => void;
  setLanguage: (lang: 'fr' | 'en') => void;
  resetAllProgress: () => void;
}

export const useAppStore = create<AppState>()(
  persist(
    (set, get) => ({
      // Default state
      selectedPreset: 'bluejay',
      imageUrl: '/presets/bluejay.jpg',
      imageFilename: 'bluejay.jpg',
      customImageData: null,
      preprocessMode: 'resize_crop_256',
      patchSize: 16,

      selectedModel: 'vit',
      diffLevel: 'miniature',

      vitResult: null,
      deitResult: null,
      isInferring: false,
      backendConnected: false,

      score: 0,
      attempts: 0,
      validatedTheorySteps: {},
      validatedPracticeSteps: {},
      badges: INITIAL_BADGES,
      studentName: 'Étudiant IA',

      theme: 'light',
      language: 'fr',

      // Actions
      selectPreset: (presetId: string) => {
        const found = PRESET_IMAGES.find((p) => p.id === presetId);
        if (found) {
          set({
            selectedPreset: presetId,
            imageUrl: found.url,
            imageFilename: found.filename,
            customImageData: null,
          });
        }
      },

      setCustomImage: (dataUrl: string, filename: string) => {
        set({
          selectedPreset: 'custom',
          imageUrl: dataUrl,
          imageFilename: filename,
          customImageData: dataUrl,
        });
      },

      setPreprocessMode: (mode) => set({ preprocessMode: mode }),
      setPatchSize: (size) => set({ patchSize: size }),
      setSelectedModel: (model) => set({ selectedModel: model }),
      setDiffLevel: (level) => set({ diffLevel: level }),

      setInferenceResults: (vit, deit) => set({ vitResult: vit, deitResult: deit }),
      setIsInferring: (status) => set({ isInferring: status }),
      setBackendConnected: (status) => set({ backendConnected: status }),

      validateTheoryStep: (stepId: number, points = 15) => {
        const currentValidated = get().validatedTheorySteps;
        if (!currentValidated[stepId]) {
          const nextValidated = { ...currentValidated, [stepId]: true };
          const nextScore = Math.min(100, get().score + points);
          set({ validatedTheorySteps: nextValidated, score: nextScore });

          // Check badge unlocking
          const badgeMap: Record<number, string> = {
            1: 'patch_pioneer',
            2: 'linear_projectionist',
            3: 'token_architect',
            4: 'norm_master',
            5: 'attention_seeker',
            6: 'gelu_activator',
            7: 'classifier_champion',
          };
          const badgeId = badgeMap[stepId];
          if (badgeId) {
            get().unlockBadge(badgeId);
          }

          // Check if all 7 theory steps completed
          if (Object.keys(nextValidated).length >= 7) {
            get().unlockBadge('grand_master');
          }
        }
      },

      validatePracticeStep: (stepId: number) => {
        const current = get().validatedPracticeSteps;
        if (!current[stepId]) {
          const next = { ...current, [stepId]: true };
          set({ validatedPracticeSteps: next });
          get().unlockBadge('practice_explorer');
        }
      },

      registerAttempt: () => set((state) => ({ attempts: state.attempts + 1 })),

      unlockBadge: (badgeId: string) => {
        set((state) => ({
          badges: state.badges.map((b) =>
            b.id === badgeId && !b.unlocked
              ? { ...b, unlocked: true, unlockedAt: new Date().toISOString() }
              : b
          ),
        }));
      },

      setStudentName: (name: string) => set({ studentName: name }),

      toggleTheme: () =>
        set((state) => {
          const nextTheme = state.theme === 'light' ? 'dark' : 'light';
          if (nextTheme === 'dark') {
            document.documentElement.classList.add('dark');
          } else {
            document.documentElement.classList.remove('dark');
          }
          return { theme: nextTheme };
        }),

      setLanguage: (lang) => set({ language: lang }),

      resetAllProgress: () => {
        set({
          score: 0,
          attempts: 0,
          validatedTheorySteps: {},
          validatedPracticeSteps: {},
          badges: INITIAL_BADGES,
        });
      },
    }),
    {
      name: 'vit-deit-learning-studio-v2',
      partialize: (state) => ({
        selectedPreset: state.selectedPreset,
        imageUrl: state.imageUrl,
        imageFilename: state.imageFilename,
        preprocessMode: state.preprocessMode,
        patchSize: state.patchSize,
        selectedModel: state.selectedModel,
        diffLevel: state.diffLevel,
        score: state.score,
        attempts: state.attempts,
        validatedTheorySteps: state.validatedTheorySteps,
        validatedPracticeSteps: state.validatedPracticeSteps,
        badges: state.badges,
        studentName: state.studentName,
        theme: state.theme,
        language: state.language,
      }),
    }
  )
);

