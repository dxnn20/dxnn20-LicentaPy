import { Routes } from '@angular/router';
import { LoginPageComponent} from "./pages/login-page/login-page.component";
import { SignupPageComponent } from './pages/signup-page/signup-page.component';
import {DashboardPageComponent} from './pages/dashboard-page/dashboard-page.component';

export const routes: Routes = [
  { path: '', redirectTo: 'login', pathMatch: 'full' },
  { path: 'login' , component: LoginPageComponent },
  { path: 'sign-up' , component: SignupPageComponent },
  { path: 'dashboard' , component: DashboardPageComponent },
];
