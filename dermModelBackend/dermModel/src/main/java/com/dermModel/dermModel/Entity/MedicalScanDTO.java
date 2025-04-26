package com.dermModel.dermModel.Entity;

import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.databind.ObjectMapper;
import lombok.Data;

import java.util.Base64;
import java.util.List;
import java.util.Map;

@Data
public class MedicalScanDTO {
    private Long   id;
    private String fileName;
    private String beforePicBase64;
    private String afterPicBase64;
    private Integer predictedClass;
    private Float   confidence;
    private List<Map<String,Object>> top3;

    public MedicalScanDTO(MedicalScan scan) throws JsonProcessingException {
        this.id               = scan.getId();
        this.fileName         = scan.getFileName();
        this.beforePicBase64  = Base64.getEncoder().encodeToString(scan.getBeforePic());
        this.afterPicBase64   = Base64.getEncoder().encodeToString(scan.getAfterPic());
        this.predictedClass   = scan.getPredictedClass();
        this.confidence       = scan.getConfidence();
        this.top3             = new ObjectMapper()
                .readValue(scan.getTop3PredictionsJson(),
                        new TypeReference<>(){});
    }
}
