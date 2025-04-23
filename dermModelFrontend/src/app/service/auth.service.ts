import { Injectable } from '@angular/core';
import { HttpClient, HttpParams } from '@angular/common/http';
import { BehaviorSubject, Observable } from 'rxjs';
import { User } from '../models/User';
import {environment} from "../../environment";
import { StorageService } from './storage.service';

@Injectable({ providedIn: 'root' })
export class AuthService {
  // Holds the latest user (or null)
  private currentUserSubject = new BehaviorSubject<User | null>(null);
  // Exposed as Observable for components to subscribe
  public currentUser$ = this.currentUserSubject.asObservable();

  constructor(private http: HttpClient, private storageService: StorageService) {
    const saved = this.storageService.getItem('user');
    if (saved) {
      this.currentUserSubject.next(JSON.parse(saved));
    }
  }

  logout(): void {
    this.storageService.removeItem('user');
    this.currentUserSubject.next(null);
  }

  public get currentUserValue(): User | null {
    return this.currentUserSubject.value;
  }

  login(email: string, password: string) {
    const body = new URLSearchParams();
    body.set('email', email);
    body.set('password', password);
    return this.http.post(
      environment.apiPrefix + '/login',
      body.toString(),
      { headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, responseType: 'text', withCredentials: true }
    );
  }

  updateCurrentUser(user: User): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('user', JSON.stringify(user));
    }
    this.currentUserSubject.next(user);
  }

}
