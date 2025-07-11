import { Component } from '@angular/core';
import {User} from "../../models/User";
import { environment } from "../../../environment";
import { HttpClient } from '@angular/common/http';
import { Router } from '@angular/router';
import {FormsModule} from "@angular/forms";
import {NgIf} from "@angular/common";

@Component({
  selector: 'app-signup-page',
  standalone: true,
  imports: [
    FormsModule,
    NgIf
  ],
  templateUrl: './signup-page.component.html',
  styleUrl: './signup-page.component.scss'
})
export class SignupPageComponent {

  constructor(private http: HttpClient, private router: Router) {
    this.user = {
      firstName: '',
      lastName: '',
      username: '',
      roles: 'PATIENT',
      email: '',
      password: '',
    };
  }

  user: User = {
    firstName: '',
    lastName: '',
    username: '',
    roles: 'PATIENT',
    email: '',
    password: '',
  };

  errorMessage: string = '';

  showSuccess: boolean = false;

  onSubmit(): void {
    this.http.post(`${environment.apiPrefix}/security/sign-up`, this.user).subscribe({
      next: (response) => {
        console.log('Signup successful:', response);
        this.showSuccess = true;

        // Hide snackbar after 2 seconds
        setTimeout(() => {
          this.showSuccess = false;
          this.router.navigate(['/login']);
        }, 2000);
      },
      error: (error) => {
        if (error.status === 409) {
          this.errorMessage = error.error?.error || 'User already exists';
        } else if (error.status === 500) {
          this.errorMessage = error.error?.error || 'Server error during registration';
        } else {
          this.errorMessage = 'Unknown error occurred';
        }

        console.error('Signup error:', error);
      },
    });
  }

}
