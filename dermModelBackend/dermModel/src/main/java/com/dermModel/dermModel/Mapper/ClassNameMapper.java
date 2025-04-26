package com.dermModel.dermModel.Mapper;

import com.fasterxml.jackson.databind.ObjectMapper;
import jakarta.annotation.PostConstruct;
import org.springframework.stereotype.Component;

import java.io.InputStream;
import java.util.Map;

@Component
public class ClassNameMapper {
    private Map<Integer, String> classMap;

    @PostConstruct
    public void init() throws Exception {
        ObjectMapper mapper = new ObjectMapper();
        try (InputStream is = getClass().getResourceAsStream("/class_map.json")) {
            classMap = mapper.readValue(is, mapper.getTypeFactory()
                    .constructMapType(Map.class, Integer.class, String.class));
        }
    }

    public String getClassName(int classId) {
        return classMap.getOrDefault(classId, "Unknown");
    }
}
