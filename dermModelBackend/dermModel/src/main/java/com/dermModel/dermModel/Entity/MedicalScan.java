package com.dermModel.dermModel.Entity;

import jakarta.persistence.*;
import lombok.*;

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

    public Long getId() {
        return id;
    }

    public void setId(Long id) {
        this.id = id;
    }

    public void setFileName(String fileName) {
        this.fileName = fileName;
    }

    public void setUser(User user) {
        this.user = user;
    }

    public void setBeforePic(byte[] beforePic) {
        this.beforePic = beforePic;
    }

    public void setAfterPic(byte[] afterPic) {
        this.afterPic = afterPic;
    }

    public void setPredictedClass(Integer predictedClass) {
        this.predictedClass = predictedClass;
    }

    public void setConfidence(Float confidence) {
        this.confidence = confidence;
    }

    public void setTop3PredictionsJson(String top3PredictionsJson) {
        this.top3PredictionsJson = top3PredictionsJson;
    }

    public void setBeforePicBase64(String beforePicBase64) {
        this.beforePicBase64 = beforePicBase64;
    }

    public void setAfterPicBase64(String afterPicBase64) {
        this.afterPicBase64 = afterPicBase64;
    }

    public void setPredictedClassName(String predictedClassName) {
        this.predictedClassName = predictedClassName;
    }

    public void setTop3(PredictionResult top3) {
        this.top3 = top3;
    }

    public String getFileName() {
        return fileName;
    }

    public User getUser() {
        return user;
    }

    public byte[] getBeforePic() {
        return beforePic;
    }

    public byte[] getAfterPic() {
        return afterPic;
    }

    public Integer getPredictedClass() {
        return predictedClass;
    }

    public Float getConfidence() {
        return confidence;
    }

    public String getTop3PredictionsJson() {
        return top3PredictionsJson;
    }

    public String getBeforePicBase64() {
        return beforePicBase64;
    }

    public String getAfterPicBase64() {
        return afterPicBase64;
    }

    public String getPredictedClassName() {
        return predictedClassName;
    }

    public PredictionResult getTop3() {
        return top3;
    }

    @Transient
    private String afterPicBase64;

    @Column(name = "predicted_class_name")
    private String predictedClassName;

    @Transient
    private PredictionResult top3;            // helper object (see DTO below)
}