export interface PresetImage {
  id: string;
  name: string;
  category: string;
  url: string;
  expectedClass: string;
  description: string;
}

export type ModelType = 'vit' | 'deit';

export interface StepDefinition {
  id: number;
  title: string;
  shortTitle: string;
  subtitle: string;
  badge: string;
  description: string;
}

export const PRACTICE_STEPS: StepDefinition[] = [
  {
    id: 1,
    title: 'Importer une image',
    shortTitle: '1. Image',
    subtitle: 'Acquisition, Redimensionnement & Normalisation',
    badge: 'PRÉTRAITEMENT',
    description: 'Importation d’une image, lecture des dimensions d’origine, redimensionnement 224×224 px et normalisation ImageNet.'
  },
  {
    id: 2,
    title: 'Découpage en patches',
    shortTitle: '2. Patches',
    subtitle: 'Grille 14×14 & Slicing 16×16 px',
    badge: 'DÉCOUPAGE',
    description: 'Découpage de l’image 2D en 196 patches de 16×16 pixels avec sélection interactive et agrandissement.'
  },
  {
    id: 3,
    title: 'Patch Embedding',
    shortTitle: '3. Projection',
    subtitle: 'Aplatissement 2D → 1D & Projection Linéaire',
    badge: 'EMBEDDING',
    description: 'Transformation de chaque patch en vecteur 768D via la projection linéaire apprenable E.'
  },
  {
    id: 4,
    title: 'Position Embedding et Tokens',
    shortTitle: '4. Tokens & PosEnc',
    subtitle: 'Ajout de [CLS], [DIST] & Encodage Positionnel 1D',
    badge: 'SÉQUENCE',
    description: 'Insertion du token CLS (et DIST pour DeiT), ajout des Position Embeddings 1D et constitution de la séquence finale.'
  },
  {
    id: 5,
    title: 'Self-Attention',
    shortTitle: '5. Attention',
    subtitle: 'Calcul Q, K, V & Extraction Couche / Tête',
    badge: 'ATTENTION',
    description: 'Calcul du mécanisme d’attention avec sélection de la couche (1-12) et de la tête (1-12) sur les poids réels du modèle.'
  },
  {
    id: 6,
    title: 'Transformer Encoder',
    shortTitle: '6. Blocs Encodeurs',
    subtitle: '12 Blocs Pre-LN, Résidus & MLP avec GELU',
    badge: 'ENCODEUR',
    description: 'Traversée des 12 blocs Transformer, normalisation par couche, connexions résiduelles et réseau feed-forward MLP.'
  },
  {
    id: 7,
    title: 'Prédiction finale',
    shortTitle: '7. Classification',
    subtitle: 'Inférence Réelle ViT vs DeiT & Top-5 Probabilités',
    badge: 'PRÉDICTION',
    description: 'Extraction des tokens de décision, têtes de classification, probabilités Top-5 et comparaison ViT vs DeiT.'
  }
];

export const PRESET_IMAGES: PresetImage[] = [
  {
    id: 'samoyed',
    name: 'Chien Samoyède',
    category: 'Canidé',
    url: '/presets/samoyed.jpg',
    expectedClass: 'Samoyed, Samoyede',
    description: 'Chien arctique blanc à fourrure dense et sourire distinctif.'
  },
  {
    id: 'car',
    name: 'Voiture de Sport',
    category: 'Véhicule',
    url: '/presets/car.jpg',
    expectedClass: 'sports car, sport car',
    description: 'Voiture sportive profilée rouge vif aérodynamique.'
  },
  {
    id: 'cat',
    name: 'Chat Tigré',
    category: 'Félin',
    url: '/presets/cat.jpg',
    expectedClass: 'tabby, tabby cat',
    description: 'Chat tigré domestique aux yeux perçants.'
  },
  {
    id: 'shark',
    name: 'Requin Tigre',
    category: 'Faune Marine',
    url: '/presets/shark.jpg',
    expectedClass: 'tiger shark, Galeocerdo cuvieri',
    description: 'Grand prédateur marin sous les reflets de l’eau.'
  },
  {
    id: 'espresso',
    name: 'Café Espresso',
    category: 'Objet / Boisson',
    url: '/presets/espresso.jpg',
    expectedClass: 'espresso',
    description: 'Tasse de café espresso fumante avec mousse onctueuse.'
  }
];
