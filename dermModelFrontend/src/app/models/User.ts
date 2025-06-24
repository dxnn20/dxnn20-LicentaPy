import { MedicalScan } from "./MedicalScan";

export interface User {
  id?: number;
  firstName: string;
  lastName: string;
  email: string;
  username: string;
  password: string;
  roles: string;
  scans?: MedicalScan[];
}
