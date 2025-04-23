import { Component, Inject, PLATFORM_ID } from '@angular/core';
import {Router, RouterOutlet} from "@angular/router";
import { AuthService } from '../../service/auth.service';
import {User} from "../../models/User";
import {FormsModule} from "@angular/forms";
import {isPlatformBrowser, NgIf} from "@angular/common";
// import { AuthService } from '../../service/auth.service';

@Component({
  selector: 'app-login-page',
  standalone: true,
  imports: [
    RouterOutlet,
    FormsModule,
    NgIf
  ],
  templateUrl: './login-page.component.html',
  styleUrl: './login-page.component.scss'
})
export class LoginPageComponent {
  user: User = {
    firstName: '',
    lastName: '',
    username: '',
    roles: 'PATIENT',
    email: '',
    password: '',
  }

  protected errorMessage: string = '';

  constructor(
    private authService: AuthService,
    private router: Router,
    @Inject(PLATFORM_ID) private platformId: Object
  ) {}

  onSubmit(): void {
    this.authService.login(this.user.email, this.user.password).subscribe({
      next: (res) => {
        console.log('Login successful:', res);

        try {
          // Parse the response if it's a string
          let userData: User;
          userData = JSON.parse(res);

          // Store user data in localStorage
          if (isPlatformBrowser(this.platformId)) {
            localStorage.setItem('user', JSON.stringify(userData));
          }

          // Update the auth service's current user
          this.authService.updateCurrentUser(userData);

          // Navigate to dashboard
          this.router.navigate(['/dashboard']);
        } catch (e) {
          console.error('Error processing login response:', e);
          this.errorMessage = 'Invalid response from server';
        }
      },
      error: (err) => {
        console.log('Login failed:', err);
        try {
          const parsed = JSON.parse(err.error);
          this.errorMessage = parsed.error || 'Something went wrong';
        } catch {
          this.errorMessage = 'Something went wrong';
        }
      }
    });
  }
}
