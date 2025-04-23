package com.dermModel.dermModel.Entity;

import jakarta.persistence.*;
import lombok.*;

import java.util.Arrays;

@Setter
@Getter
@Entity
@Table(name = "medical_scans")
@Data
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MedicalScan {
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    private String fileName;

    @ManyToOne(fetch = FetchType.EAGER)
    @JoinColumn(name = "user_id", nullable = false)
    private User user;

    @Lob
    @Column(name = "before_pic", columnDefinition = "LONGBLOB")
    private byte[] beforePic;

    @Lob
    @Column(name = "after_pic", columnDefinition = "LONGBLOB")
    private byte[] afterPic;

    private Integer predictedClass;           // index of top‑1
    private Float   confidence;               // probability of predictedClass

    @Column(length = 1024)
    private String  top3PredictionsJson;      // JSON string [{"class":5,"conf":0.87},…]

    // Transient fields for returning in DTO
    @Transient
    private String beforePicBase64;

    @Transient
    private String afterPicBase64;

    @Column(name = "predicted_class_name")
    private String predictedClassName;

    @Transient
    private PredictionResult top3;            // helper object (see DTO below)
}