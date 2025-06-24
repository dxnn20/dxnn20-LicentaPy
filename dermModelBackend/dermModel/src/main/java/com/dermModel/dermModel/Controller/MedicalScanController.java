package com.dermModel.dermModel.Controller;

import com.dermModel.dermModel.Entity.MedicalScan;
import com.dermModel.dermModel.Entity.User;
import com.dermModel.dermModel.Mapper.ClassNameMapper;
import com.dermModel.dermModel.Repository.MedicalScanRepository;
import com.dermModel.dermModel.Repository.UserRepository;
import com.dermModel.dermModel.Service.MedicalScanService;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.security.core.Authentication;
import org.springframework.security.core.context.SecurityContextHolder;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.server.ResponseStatusException;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.*;

@RestController
@RequestMapping("/medical-scans")
public class MedicalScanController {

    @Autowired
    private ClassNameMapper classNameMapper;

    @Autowired
    MedicalScanService medicalScanService;

    @Autowired
    MedicalScanRepository medicalScanRepository;

    @Autowired
    UserRepository userRepository;

    @GetMapping("/get-all-by-user-id/{id}")
    public List<MedicalScan> getAllByUserId(@PathVariable Long id) {
        List<MedicalScan> scans = medicalScanService.getMedicalScansByUserId(id);

        // Convert images to Base64 format just before sending them to the frontend
        for (MedicalScan scan : scans) {
            if (scan.getBeforePic() != null) {
                String base64BeforePic = Base64.getEncoder().encodeToString(scan.getBeforePic());
                scan.setBeforePicBase64(base64BeforePic);  // Set the Base64 string in a field
            }
            if (scan.getAfterPic() != null) {
                String base64AfterPic = Base64.getEncoder().encodeToString(scan.getAfterPic());
                scan.setAfterPicBase64(base64AfterPic);  // Set the Base64 string in a field
            }
        }
        return scans;
    }

    @PostMapping(value = "/save-scan", consumes = MediaType.MULTIPART_FORM_DATA_VALUE)
    public ResponseEntity<?> saveScan(@RequestParam("file") MultipartFile file) {
        try {
            // Retrieve authenticated user
            Authentication auth = SecurityContextHolder.getContext().getAuthentication();
            String principalJson = auth.getName();

            ObjectMapper mapper = new ObjectMapper();
            Map<String, Object> userMap = mapper.readValue(principalJson, Map.class);
            String email = (String) userMap.get("email");  // extract email from JSON

            System.out.println("User email: " + email);

            // Now we can use the email to retrieve the user from the DB
            User user = userRepository.findByEmail(email).orElseThrow();

            // Create and save scan
            MedicalScan scan = new MedicalScan();
            scan.setFileName(file.getOriginalFilename());
            scan.setBeforePic(file.getBytes());
            scan.setUser(user);

            medicalScanService.addMedicalScan(scan);

            return ResponseEntity.ok(Collections.singletonMap("message", "Success"));
        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(500).body("Upload failed: " + e.getMessage());
        }
    }

    @DeleteMapping(value = "/delete-scan/{id}")
    public ResponseEntity<?> deleteScan(@PathVariable("id") Long id) {
        try {
            // Get the authenticated user
            Authentication auth = SecurityContextHolder.getContext().getAuthentication();
            ObjectMapper mapper = new ObjectMapper();
            User authenticatedUser = mapper.readValue(auth.getName(), User.class);

            Optional<MedicalScan> scanOptional = medicalScanRepository.findById(id);
            if (scanOptional.isEmpty()) {
                return ResponseEntity.status(HttpStatus.NOT_FOUND).body("Scan not found");
            }

            MedicalScan scan = scanOptional.get();

            // Check that the scan belongs to the authenticated user
            if (!scan.getUser().getId().equals(authenticatedUser.getId())) {
                return ResponseEntity.status(HttpStatus.FORBIDDEN).body("You are not authorized to delete this scan");
            }

            System.out.println("Deleting scan with id: " + id);

            // Delete the scan
            medicalScanRepository.delete(scan);
            return ResponseEntity.ok(Collections.singletonMap("message", "Success"));

        } catch (Exception e) {
            e.printStackTrace();
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body("Error deleting scan");
        }
    }

    @PostMapping(value = "/update-scan")
    public ResponseEntity<MedicalScan> updateScan(
            @RequestBody Map<String, Long> request
    ) throws Exception {

        Long scanId = request.get("scanId");

        System.out.println("Updating scan with id: " + scanId);

        // 0) Fetch the existing scan (throws 404 if not found)
        MedicalScan scan = medicalScanRepository.findById(scanId)
                .orElseThrow(() -> new ResponseStatusException(HttpStatus.NOT_FOUND, "Scan not found"));

        byte[] before = scan.getBeforePic();

        // 2) Write 'before' to a temp file for the Grad‑CAM script
        Path tmpInput = Files.createTempFile("scan-", scan.getFileName());
        Path tmpOutput = Files.createTempFile("overlay-", ".png");
        Files.write(tmpInput, before);

        System.out.println(tmpInput);
        System.out.println(tmpOutput);

        System.out.println("Current working directory: " + System.getProperty("user.dir"));

        try {
            // 3) Call the Python script
            ProcessBuilder pb = new ProcessBuilder(
                    "D:\\LICENTAPY\\.venv\\Scripts\\python.exe",
                    "D:\\LICENTAPY\\dermModelBackend\\dermModel\\src\\predict_and_gradcam.py",
                    "--input", tmpInput.toString(),
                    "--output", tmpOutput.toString()
            );

            pb.redirectErrorStream(true);  // Redirect both stdout and stderr to the same stream, need to see what it outputs
            Process proc = pb.start();

            // Capture the output and error streams from the Python process
            BufferedReader reader2 = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line;
            StringBuilder output = new StringBuilder();
            while ((line = reader2.readLine()) != null) {
                output.append(line).append("\n");
            }
            proc.waitFor();  // Wait for the process to finish

            // Log the output
            System.out.println("Python script output: " + output.toString());

// Handle script failure
            if (proc.exitValue() != 0) {
                throw new RuntimeException("GradCAM script failed or timed out");
            }

            // read JSON from stdout
            String jsonString;
            try (BufferedReader _ = new BufferedReader(
                    new InputStreamReader(proc.getInputStream()))) {
                jsonString = output.toString().trim();
            }

            ObjectMapper mapper = new ObjectMapper();
            JsonNode json = mapper.readTree(jsonString);

            // 4) Read the overlay bytes
            byte[] after = Files.readAllBytes(tmpOutput);

            // 5) Update prediction fields on the existing entity
            scan.setAfterPic(after);

            JsonNode predNode = json.get("predicted_class");
            scan.setPredictedClass(predNode != null ? predNode.asInt() : -1);
            if (predNode != null) {
                int predictedClassId = predNode.asInt();
                String predictedClassName = classNameMapper.getClassName(predictedClassId);
                scan.setPredictedClassName(predictedClassName);
            }

            JsonNode confNode = json.get("confidence");
            scan.setConfidence(confNode != null ? (float) confNode.asDouble() : 0.0f);

            JsonNode top3Node = json.get("top3");
            if (top3Node != null && top3Node.isArray()) {
                scan.setTop3PredictionsJson(top3Node.toString());
            } else {
                scan.setTop3PredictionsJson("[]"); // fallback to empty array
            }

            // 6) Persist the updated scan
            MedicalScan updated = medicalScanRepository.save(scan);

            // 7) Return the DTO
            return ResponseEntity.ok(updated);
        } finally {
            // 8) Clean up temp files in all cases
            Files.deleteIfExists(tmpInput);
            Files.deleteIfExists(tmpOutput);
        }
    }
}
