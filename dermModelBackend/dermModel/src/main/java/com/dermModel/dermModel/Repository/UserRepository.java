package com.dermModel.dermModel.Repository;

import com.dermModel.dermModel.Entity.MedicalScan;
import com.dermModel.dermModel.Entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;

import java.util.List;
import java.util.Optional;

public interface UserRepository extends JpaRepository<User, Long> {

    @Query("select m from MedicalScan m where User = :userId")
    List<MedicalScan> getMedicalScans(Long userId);

    @Query(value="select user from User user where user.email = ?1")
    Optional<User> findByEmail(String email);

}
