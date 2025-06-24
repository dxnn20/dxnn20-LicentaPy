import {Component, OnInit} from '@angular/core';
import {Router} from '@angular/router';
import {AuthService} from '../../service/auth.service';
import {ScanService} from '../../service/scan.service';
import {User} from '../../models/User';
import {NgForOf, NgIf, NgOptimizedImage, PercentPipe} from "@angular/common";
import {MedicalScan} from "../../models/MedicalScan";

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [
    NgIf,
    NgForOf,
    PercentPipe,
    NgOptimizedImage,
  ],
  templateUrl: './dashboard-page.component.html',
  styleUrls: ['./dashboard-page.component.scss'],
})
export class DashboardPageComponent implements OnInit {
  user: User | null = null;
  scans: any[] = [];
  selectedFile: File | null = null;
  selectedScan: MedicalScan | null = null;

  constructor(
    private authService: AuthService,
    private scanService: ScanService,
    protected router: Router
  ) {}

  openModal(scan: MedicalScan) {
    this.selectedScan = scan;
    const modal: HTMLInputElement | null = document.getElementById('scan-modal') as HTMLInputElement;
    if (modal) modal.checked = true;

    this.loadScans()
  }

  getOverlaySrc(scan: MedicalScan): string {
    return `data:image/png;base64,${scan.afterPicBase64}`;
  }

  ngOnInit(): void {
    this.user = this.authService.currentUserValue;
    if (!this.user) {
      // if no user, redirect to login
      this.router.navigate(['/login']);
      return;
    }
    this.loadScans();
  }

  logout(): void {
    this.authService.logout();
    this.router.navigate(['/login']);
  }

  onFileSelected(e: Event) {
    this.selectedFile = (e.target as HTMLInputElement).files?.[0] ?? null;
  }

  uploadScan() {
    if (!this.selectedFile || !this.user) return;
    this.scanService.uploadScan(this.selectedFile)
      .subscribe(() => this.loadScans());

    this.selectedFile = null; // Reset the file input after upload
    this.selectedScan = null; // Reset the selected scan
  }

  deleteScan(id: number) {
    if (!this.user) return;
    this.scanService.deleteScan(id)
      .subscribe(() => this.loadScans());

  }

  loadScans() {
    if (!this.user) return;
    this.scanService.getScansByUser(this.user.id)
      .subscribe(data => {
        this.scans = data;
      });
  }

  getScanSrc(scan: any): string {
    return `data:image/png;base64,${scan.beforePicBase64}`;
  }

  processScan(id: number) {
    if (!this.user) return;

    console.log('Processing scan with ID:', id);

    this.scanService.processScan(id).subscribe({
      next: (response) => {
        console.log('Scan processed:', response);
        this.loadScans();
      },
      error: (error) => {
        console.error('Error processing scan:', error);
      }
    });
  }

  get top3PredictedLabels(): string[] {
    try {
      const ids: number[] = JSON.parse(this!.selectedScan!.top3PredictionsJson || '[]');
      return ids.map(id => this.convertToClass(id));
    } catch (e) {
    console.error('Invalid top3PredictionsJson:', this!.selectedScan!.top3PredictionsJson);
      return [];
    }
  }

  convertToClass(classNr: number): string {
    const classMap: { [key: number]: string } = {
      "0": "Acne and Rosacea Photos",
      "1": "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
      "2": "Exanthems and Drug Eruptions",
      "3": "Hair Loss Photos Alopecia and other Hair Diseases",
      "4": "Melanoma Skin Cancer Nevi and Moles",
      "5": "Nail Fungus and other Nail Disease",
      "6": "Psoriasis pictures Lichen Planus and related diseases",
      "7": "Seborrheic Keratoses and other Benign Tumors",
      "8": "Tinea Ringworm Candidiasis and other Fungal Infections",
      "9": "Vascular Tumors"
    }
    return classMap[classNr] || "Unknown Class";
  }

}
