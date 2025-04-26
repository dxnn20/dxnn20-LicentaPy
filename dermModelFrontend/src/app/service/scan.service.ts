import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import {MedicalScan} from "../models/MedicalScan";
import {environment} from "../../environment";


@Injectable({
  providedIn: 'root'
})
export class ScanService {

  constructor(protected http: HttpClient) { }

  getScansByUser(id: number | undefined) {
    return this.http.get<MedicalScan[]>(`${environment.apiPrefix}/medical-scans/get-all-by-user-id/${id}`); }
  uploadScan(file: File) {
    const form = new FormData();
    form.append('file', file);
    return this.http.post(
      `${environment.apiPrefix}/medical-scans/save-scan`,
      form,
      { withCredentials: true }
    );
  }
  deleteScan(id: number) {
    return this.http.delete(`${environment.apiPrefix}/medical-scans/delete-scan/${id}`, { withCredentials: true });
  }

  processScan(scanId: number) {
    return this.http.post(
      `${environment.apiPrefix}/medical-scans/update-scan`,
      {'scanId': scanId},
      { withCredentials: true}
    );
  }
}
