SmartDerm este o aplicație de tip full-stack care utilizează inteligența artificială pentru diagnosticarea precisă a afecțiunilor cutanate. Sistemul combină performanțele ridicate ale arhitecturii EfficientNet-B5 cu explicabilitatea oferită de Grad-CAM, oferind utilizatorilor nu doar o predicție, ci și o justificare vizuală a diagnosticului.

Caracteristici Principale
Model AI Performant: EfficientNet-B5 cu 95.96% acuratețe top-3

AI Explicabil: Implementare Grad-CAM pentru vizualizarea regiunilor relevante

Interfață Web Modernă: Frontend Angular cu backend Spring Boot

10 Clase de Afecțiuni: Detectarea a 10 tipuri distincte de boli de piele

Procesare Rapidă: Timp de răspuns sub 3 secunde

Arhitectură Scalabilă: Separare front-end/back-end pentru flexibilitate

SmartDerm/
├── dermModelBackend/           # Backend Spring Boot
│   └── dermModel/             # Aplicația principală
│       ├── src/               # Codul sursă Java
│       ├── pom.xml           # Dependințe Maven
│       └── target/           # Fișiere compilate
├── dermModelFrontend/         # Frontend Angular
│   ├── src/                  # Codul sursă TypeScript/Angular
│   ├── package.json         # Dependințe npm
│   └── dist/                # Build de producție
├── ModelCode.py              # Cod de antrenare PyTorch
├── CreateClassJSON.py        # Creare mapping clase
├── best_model_weights.pth    # Greutăți model antrenat
└── .gitignore               # Fișiere ignorate

⚙️ Cerințe de Sistem
Cerințe Hardware Minime
RAM: 8GB (16GB recomandat optional pentru antrenare)

Spațiu disc: 10GB liberi

GPU: NVIDIA GPU cu CUDA support (opțional, pentru antrenare)

CPU: Intel i5 sau echivalent AMD

Sistem de Operare Suportat
Windows 10/11 (x64)

Cerințe Software de Bază:
  Java Development Kit (JDK)
  Node.js și npm (sau un alt package manager)
  Python și pip
  MySQL Server

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

Backend:
actualizare dependinte maven (sync)
mvn spring-boot:run

Frontend:
ng serve

------------------------------------------------------------------------------------------------------------
ENGLISH
📖 About the Project
SmartDerm is a full-stack application that utilizes artificial intelligence for accurate diagnosis of skin conditions. The system combines the high performance of EfficientNet-B5 architecture with explainability provided by Grad-CAM, offering users not just a prediction, but also a visual justification of the diagnosis.

Key Features
High-Performance AI Model: EfficientNet-B5 with 95.96% top-3 accuracy

Explainable AI: Grad-CAM implementation for visualizing relevant regions

Modern Web Interface: Angular frontend with Spring Boot backend

10 Disease Classes: Detection of 10 distinct types of skin diseases

Fast Processing: Response time under 3 seconds

Scalable Architecture: Frontend/backend separation for flexibility

SmartDerm/
├── dermModelBackend/           # Spring Boot Backend
│   └── dermModel/             # Main application
│       ├── src/               # Java source code
│       ├── pom.xml           # Maven dependencies
│       └── target/           # Compiled files
├── dermModelFrontend/         # Angular Frontend
│   ├── src/                  # TypeScript/Angular source code
│   ├── package.json         # npm dependencies
│   └── dist/                # Production build
├── ModelCode.py              # PyTorch training code
├── CreateClassJSON.py        # Class mapping creation
├── best_model_weights.pth    # Trained model weights
└── .gitignore               # Ignored files

⚙️ System Requirements
Minimum Hardware Requirements
RAM: 8GB (16GB recommended for training)

Disk Space: 10GB free

GPU: NVIDIA GPU with CUDA support (optional, for training)

CPU: Intel i5 or equivalent AMD

Supported Operating Systems
Windows 10/11 (x64)

🔧 Installation and Setup
1. Basic Software Requirements
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

Backend:
update maven (sync)
mvn spring-boot:run

Frontend:
ng serve
