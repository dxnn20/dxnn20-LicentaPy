export interface MedicalScan {
  id: number;
  fileName: string;
  userId: number;
  beforePicBase64: string;
  afterPicBase64?: string;
  predictedClass?: number;
  predictedClassName?: string;
  confidence?: number;
  top3PredictionsJson?: string;
}
