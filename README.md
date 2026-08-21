NVIDIA JETSON - MICRO-EXPRESSIONS DETECTOR

Acesta este un sistem pentru detectarea micro-expresiilor faciale în timp real, combinând modelul YOLO (pentru detectarea feței) și Google MediaPipe (pentru extragerea punctelor cheie și a mișcărilor musculare).

STRUCTURA:
- main.py: Punctul de pornire al aplicației (inițializează camera).
- inferenta_colab.py: Logica de integrare și filtrare a emoțiilor.
- best.pt: Modelul YOLO antrenat manual de echipa noastră.
- Nvidia_Jetson_Project.ipynb: Mediul de lucru cu antrenarea originală din Colab.

INSTALARE SI RULARE:
1. Instalați dependențele în terminal: 
pip install ultralytics mediapipe opencv-python roboflow numpy jupyter

2. Porniți aplicația: 
python main.py (Apăsați tasta ESC pentru a închide camera).

NOTA DESPRE ISTORICUL FISIERELOR:
Pentru a menține acest repository curat și ușor de citit, fișierele principale au fost încărcate pe GitHub direct în versiunea lor finală. Logica a fost deja dezvoltată și testată complet în Google Colab, nefiind necesare modificări majore de cod în timpul asamblării pe mediul local.

Echipa: Ionescu Andrei-Cristian, Ionescu Alex Gabriel, Voicu Daria Stefania. 

----------------------------------------------------------------------

NVIDIA JETSON - MICRO-EXPRESSIONS DETECTOR

This is a real-time micro-expression detection system combining YOLO (for face detection) and Google MediaPipe (for extracting facial landmarks and muscle movements).

STRUCTURE:
- main.py: The entry point of the application (initializes the camera).
- inferenta_colab.py: Integration logic and emotion filtering system.
- best.pt: The custom YOLO model trained by our team.
- Nvidia_Jetson_Project.ipynb: The original training notebook from Colab.

INSTALLATION AND EXECUTION:
1. Install dependencies in the terminal: 
pip install ultralytics mediapipe opencv-python roboflow numpy jupyter

2. Run the application: 
python main.py (Press ESC to close the camera).

NOTE ON COMMIT HISTORY:
To keep this repository clean and easy to read, the main files were uploaded to GitHub directly in their final consolidated version. The core logic was already fully developed and tested in Google Colab, requiring no major code modifications during local assembly.

Team: Ionescu Andrei-Cristian, Ionescu Alex Gabriel, Voicu Daria Stefania. 
