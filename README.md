LINK REPO: https://github.com/dxnn20/dxnn20-LicentaPy

RO: (Scroll down for English ReadME)
SmartDerm este o aplicație de tip full-stack care utilizează inteligența artificială pentru diagnosticarea precisă a afecțiunilor cutanate. Sistemul combină performanțele ridicate ale arhitecturii EfficientNet-B5 cu explicabilitatea oferită de Grad-CAM, 
oferind utilizatorilor nu doar o predicție, ci și o justificare vizuală a diagnosticului, impreuna cu o interfata web cu o baza de date.

Caracteristici Principale:
```txt
Model AI Performant: EfficientNet-B5 cu 95.96% acuratețe top-3
AI Explicabil: Implementare Grad-CAM pentru vizualizarea regiunilor relevante
Interfață Web Modernă: Frontend Angular cu backend Spring Boot
10 Clase de Afecțiuni: Detectarea a 10 tipuri distincte de boli de piele
Procesare Rapidă: Timp de răspuns foarte rapid si pe hardware limitat
Arhitectură Scalabilă: Separare front-end/back-end pentru flexibilitate
```
Sistem de Operare Suportat
Windows 10/11 (x64)
```
Cerințe Software de Bază:
  Java Development Kit (JDK)
  Node.js și npm (sau un alt package manager)
  Python și pip
  MySQL Server
```

```
instalare dependinte Python:
pip install torch
pip install pytorch-gradcam
pip install Pillow
pip install numpy
pip install opencv-python
pip install matplotlib
pip install scikit-learn
pip install tqdm
pip install onnx
pip install onnxruntime
```

Rulare Backend:
actualizare dependinte maven (sync)
mvn spring-boot:run

Rulare Frontend:
ng serve

IMPORTANT!
Pentru hardware diferit poate fi necesar schimbarea porturilor, a parolei pentru conectarea la baza de date, sau alte specificatii ce pot fi diferite in functie de !

------------------------------------------------------------------------------------------------------------
ENGLISH
About the Project
SmartDerm is a full-stack application that utilizes artificial intelligence for accurate diagnosis of skin conditions. The system combines the high performance of EfficientNet-B5 architecture with explainability provided by Grad-CAM, offering users not just a prediction, but also a visual justification of the diagnosis.

Key Features:
High-Performance AI Model: EfficientNet-B5 with 95.96% top-3 accuracy
Explainable AI: Grad-CAM implementation for visualizing relevant regions
Modern Web Interface: Angular frontend with Spring Boot backend
10 Disease Classes: Detection of 10 distinct types of skin diseases
Fast Processing: Response time under 3 seconds
Scalable Architecture: Frontend/backend separation for flexibility

Minimum Hardware Requirements
RAM: 8GB (16GB recommended for training)
Disk Space: 10GB free
GPU: NVIDIA GPU with CUDA support (optional, for training)
CPU: Intel i5 or equivalent AMD

Supported Operating Systems
Windows 10/11 (x64)

Basic Software Requirements
  Java Development Kit (JDK)
  Node.js and npm
  Python and pip
  MySQL Server

Install Python dependencies:
pip install torch
pip install pytorch-gradcam
pip install Pillow
pip install numpy
pip install opencv-python
pip install matplotlib
pip install scikit-learn
pip install tqdm
pip install onnx
pip install onnxruntime

Run Backend:
update maven (sync)
mvn spring-boot:run

Run Frontend:
ng serve
