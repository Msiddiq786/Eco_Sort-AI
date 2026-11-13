// Shared TypeScript types for the Smart Waste Segregation app

export type WasteCategory =
  | 'cardboard'
  | 'glass'
  | 'metal'
  | 'paper'
  | 'plastic'
  | 'trash';

export type WasteCategoryV2 = WasteCategory | 'organic';

export const WASTE_CLASSES: WasteCategory[] = [
  'cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash',
];

export const V2_WASTE_CLASSES: WasteCategoryV2[] = [
  'cardboard', 'glass', 'metal', 'paper', 'plastic', 'organic', 'trash',
];

export interface ClassProbabilities {
  cardboard: number;
  glass: number;
  metal: number;
  paper: number;
  plastic: number;
  trash: number;
  organic?: number;
}

export interface RecyclingDecision {
  disposal_status: string;
  status_label: string;
  action: string;
  preparation_steps: string[];
  disposal_bin: string;
  reason: string;
  safety_info?: string | null;
  regional_caveats: string;
}

export interface PredictResponse {
  id: number;
  filename: string;
  original_filename?: string;
  image_path?: string;
  predicted_category: WasteCategoryV2;
  confidence: number;
  calibrated_confidence?: number;
  vit_prediction: string;
  clip_prediction: string;
  class_probabilities: ClassProbabilities;
  calibrated_probabilities?: ClassProbabilities;
  vit_probabilities?: ClassProbabilities;
  clip_probabilities?: ClassProbabilities;
  is_low_confidence?: boolean;
  is_ambiguous?: boolean;
  is_out_of_distribution?: boolean;
  confidence_status?: 'normal' | 'low_confidence' | 'ambiguous' | 'out_of_distribution';
  ood_warning?: string | null;
  recycling_recommendation: string;
  recycling_decision?: RecyclingDecision;
  model_agreement?: boolean | null;
  model_coverage?: ModelCoverage | null;
  model_version: string;
  created_at: string;
}

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface ModelCoverage {
  vit_supported: boolean;
  clip_supported: boolean;
  ensemble_supported: boolean;
  unsupported_reason?: string | null;
  compatibility_decision?: string | null;
  coverage_type?: 'full' | 'partial' | null;
}

export interface DetectedObject {
  object_id: number;
  bbox: BoundingBox;
  detection_confidence: number;
  detection_label: string;
  material_category: string;
  vit_prediction: string;
  clip_prediction: string;
  final_category: string;
  class_probabilities: Record<string, number>;
  calibrated_confidence: number;
  confidence_status: string;
  is_out_of_distribution?: boolean;
  model_agreement?: boolean;
  model_coverage?: ModelCoverage;
  crop_width?: number;
  crop_height?: number;
  recycling_status?: string;
  disposal_recommendation?: string;
  preparation_steps?: string[];
  recycling_decision: RecyclingDecision;
}

export interface RecyclingSummary {
  total_objects: number;
  category_counts: Record<string, number>;
  status_counts: {
    recyclable: number;
    conditionally_recyclable: number;
    compostable: number;
    hazardous: number;
    not_standard: number;
    unknown: number;
  };
}

export interface DetectResponse {
  image_width: number;
  image_height: number;
  total_objects: number;
  objects: DetectedObject[];
  category_counts: Record<string, number>;
  recycling_summary: RecyclingSummary;
  model_versions: Record<string, string>;
  inference_time_ms: number;
  annotated_image_base64: string;
  detection_model_status: string;
}

export type RecyclingPossibility =
  | 'Widely Recyclable'
  | 'Conditionally Recyclable'
  | 'Compostable'
  | 'Special Disposal Required'
  | 'Residual Waste (Non-Recyclable)';

export interface RecyclingGuideMaterial {
  id: string;
  name: string;
  category: WasteCategoryV2 | 'hazardous' | 'dropoff';
  recyclingPossibility: RecyclingPossibility;
  badgeVariant: 'success' | 'warning' | 'info' | 'danger' | 'neutral';
  description: string;
  commonExamples: string[];
  preparationInstructions: string[];
  disposalMethod: {
    binName: string;
    binColor: string;
    instructions: string;
  };
  avoidMixing: string[];
  environmentalBenefits: string;
  safetyWarnings?: string;
  iconName?: string;
}

export type AnalyticsDateFilter = 'today' | '7days' | '30days' | '90days' | 'all' | 'custom';

export interface HistoryItem {
  id: number;
  filename: string;
  original_filename?: string;
  image_path?: string;
  predicted_category: WasteCategoryV2 | string;
  confidence: number;
  vit_prediction: string;
  clip_prediction: string;
  class_probabilities: Record<string, number>;
  created_at: string;
}

export interface PaginatedHistory {
  items: HistoryItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface CategoryStat {
  category: WasteCategoryV2 | string;
  count: number;
  percentage: number;
  is_recyclable?: boolean;
}

export interface StatsResponse {
  total_predictions: number;
  category_distribution: CategoryStat[];
  average_confidence: number;
}

export interface ModelInfoResponse {
  vit_checkpoint: string;
  clip_checkpoint: string;
  vit_model_name?: string;
  clip_model_name?: string;
  clip_enabled?: boolean;
  ensemble_vit_weight: number;
  ensemble_clip_weight: number;
  waste_classes: WasteCategory[];
  model_version: string;
  vit_loaded: boolean;
  clip_loaded: boolean;
}

export interface CLIPPredictResponse {
  predicted_category: WasteCategory;
  confidence: number;
  probabilities: ClassProbabilities;
}

export interface HealthResponse {
  status: string;
  vit_loaded: boolean;
  clip_loaded: boolean;
  models_ready: boolean;
}

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface ModelMetrics {
  accuracy: number;
  macro_precision: number;
  macro_recall: number;
  macro_f1: number;
  per_class: Record<WasteCategory, PerClassMetric>;
  confusion_matrix: number[][];
  class_names: WasteCategory[];
}

export interface TrainLogEntry {
  epoch: number;
  train_loss: number;
  train_acc: number;
  val_loss: number;
  val_acc: number;
}

// ── Eco Learning & Quiz Types ──────────────────────────────────────────
export interface QuizQuestion {
  id: number;
  question: string;
  category: WasteCategory | 'general';
  options: string[];
  correctIndex: number;
  explanation: string;
}

export interface QuizResult {
  score: number;
  total: number;
  percentage: number;
  completedAt: string;
}

// ── Gamification & Rewards Types ───────────────────────────────────────
export interface EcoActivity {
  id: string;
  title: string;
  category: WasteCategory | 'general';
  points: number;
  timestamp: string;
  description?: string;
}

export interface DailyChallenge {
  id: string;
  title: string;
  description: string;
  points: number;
  targetCount: number;
  currentCount: number;
  completed: boolean;
  category?: WasteCategory;
}

export interface Badge {
  id: string;
  name: string;
  description: string;
  icon: string;
  category: string;
  unlockedAt: string | null;
}

export interface UserRewardState {
  points: number;
  level: number;
  streakDays: number;
  lastActiveDate: string | null;
  activities: EcoActivity[];
  unlockedBadgeIds: string[];
  completedChallengeIds: string[];
}

// ── Environmental Impact Types ────────────────────────────────────────
export interface WasteWeightLog {
  id: string;
  category: WasteCategory;
  weightGrams: number;
  loggedAt: string;
  note?: string;
}

export interface ImpactSummary {
  totalWeightKg: number;
  co2SavedKg: number;
  waterSavedLiters: number;
  treesSaved: number;
  landfillDivertedM3: number;
}
