package com.dermModel.dermModel.Service;

import com.dermModel.dermModel.Entity.MedicalScan;
import com.dermModel.dermModel.Entity.MedicalScanDTO;
import com.dermModel.dermModel.Repository.MedicalScanRepository;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.transaction.Transactional;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.server.ResponseStatusException;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.stream.Collectors;

@Service
public class MedicalScanService {
    @Autowired
    MedicalScanRepository medicalScanRepository;

    public MedicalScan addMedicalScan(MedicalScan newMedicalScan) {
        return medicalScanRepository.save(newMedicalScan);
    }

    public List<MedicalScan> getMedicalScansByUserId(Long userId) {
        return medicalScanRepository.findAllByUserId(userId);  // Use the updated method name
    }

}
