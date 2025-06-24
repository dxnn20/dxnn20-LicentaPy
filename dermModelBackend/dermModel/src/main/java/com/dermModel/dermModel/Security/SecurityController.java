package com.dermModel.dermModel.Security;

import com.dermModel.dermModel.Entity.User;
import com.dermModel.dermModel.Repository.UserRepository;
import com.dermModel.dermModel.Service.UserService;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.bcrypt.BCrypt;
import org.springframework.web.bind.annotation.*;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;

@RestController()
@RequestMapping("/security")
public class SecurityController {

    @Autowired
    private UserRepository userRepository;

    @GetMapping()
    public String security() throws JsonProcessingException
    {
        Map<String, Object> responseBody=new HashMap<>();
        responseBody.put("loggedin","yes");

        ObjectMapper mapper=new ObjectMapper();
        return mapper.writeValueAsString(responseBody);
    }

    @PostMapping("/sign-up")
    public ResponseEntity<?> securitySignUp(@RequestBody User user) {
        System.out.println("Sign Up User");

        if (UserService.userExists(user.getEmail(), userRepository)) {
            // Conflict: User already exists
            return ResponseEntity
                    .status(HttpStatus.CONFLICT)
                    .body(Collections.singletonMap("error", "User already exists!"));
        }

        try {
            user.setRoles("PATIENT");
            String salt = BCrypt.gensalt();
            user.setPassword(BCrypt.hashpw(user.getPassword(), salt));
            userRepository.save(user);

            return ResponseEntity
                    .status(HttpStatus.CREATED)
                    .body(Collections.singletonMap("message", "User successfully created!"));
        } catch (Exception e) {
            return ResponseEntity
                    .status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body(Collections.singletonMap("error", "Failed to create user!"));
        }
    }

}