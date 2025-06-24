package com.dermModel.dermModel.Service;

import com.dermModel.dermModel.Entity.User;
import com.dermModel.dermModel.Repository.UserRepository;
import org.springframework.security.crypto.bcrypt.BCrypt;
import org.springframework.stereotype.Service;

import java.util.Optional;

@Service
public class UserService {
    public User validateUser(String email, String password, UserRepository userRepository) {
        return userRepository.findByEmail(email)
                .filter(user -> BCrypt.checkpw(password, user.getPassword()))
                .orElse(null);
    }

    public static boolean userExists(String email, UserRepository userRepository)
    {
        Optional<User> user=userRepository.findByEmail(email);
        return user.isPresent();
    }

}
