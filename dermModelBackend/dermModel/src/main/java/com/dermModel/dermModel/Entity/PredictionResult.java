package com.dermModel.dermModel.Entity;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.util.List;
import java.util.Map;

// A simple DTO/helper
@Data
@AllArgsConstructor
@NoArgsConstructor
public class PredictionResult {
    private Integer predictedClass;
    private Float   confidence;
    private List<Map<String, Object>> top3;  // each map has "class" and "confidence"
}