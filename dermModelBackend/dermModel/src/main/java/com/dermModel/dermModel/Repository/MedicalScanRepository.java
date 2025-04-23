package com.dermModel.dermModel.Repository;

import com.dermModel.dermModel.Entity.MedicalScan;
import com.dermModel.dermModel.Entity.User;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.data.jpa.repository.Query;
import org.springframework.data.repository.query.Param;

import java.util.List;

public interface MedicalScanRepository extends JpaRepository<MedicalScan, Long> {

    @Query("select u from User u join u.scans s where s.id = :medicalScanId")
    User findUserByMedicalScanId(@Param("medicalScanId") Long medicalScanId);

    @Query("select scans from MedicalScan scans where scans.user.id = :id")
    List<MedicalScan> findAllByUserId(@Param("id") Long id);

}
