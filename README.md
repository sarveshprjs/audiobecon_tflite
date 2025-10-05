# audiobecon_tflite

## Project Title
audiobecon_tflite

## Description
audiobecon_tflite is a cross-platform Flutter application for real-time audio classification using TensorFlow Lite. It uses the YAMNet model (`yamnet.tflite`) to recognize environmental sounds from microphone input or audio files. The app provides instant feedback on detected sound classes and works on mobile, desktop, and web.

## Installation
1. **Clone the Repository**
    ```sh
    git clone https://github.com/yourusername/audiobecon_tflite.git
    cd audiobecon_tflite
    ```
2. **Install Flutter**
    - Follow instructions at [flutter.dev](https://flutter.dev/docs/get-started/install) for your OS.
3. **Install Dependencies**
    ```sh
    flutter pub get
    ```
4. **Add YAMNet Model and Class Map**
    - Ensure `assets/yamnet.tflite` and `assets/yamnet_class_map.csv` are present.

## Usage
- **Run on Android/iOS**
    ```sh
    flutter run
    ```
- **Run on Desktop (Windows, macOS, Linux)**
    ```sh
    flutter run -d windows   # or macos, linux
    ```
- **Run on Web**
    ```sh
    flutter run -d chrome
    ```
- The app will launch and start listening for audio input, displaying detected sound classes.

## Requirements / Dependencies
- **Languages:** Dart
- **Frameworks:** Flutter
- **Libraries:** TensorFlow Lite (via Flutter plugin), audio plugins
- **Assets:** `yamnet.tflite`, `yamnet_class_map.csv`
- **Supported Platforms:** Android, iOS, Windows, macOS, Linux, Web

## Contributing
- Fork the repository and submit pull requests.
- Follow Dart and Flutter best practices.
- Write clear commit messages and document code.

## License
No explicit license file found. Please check with the repository owner or add a license (e.g., MIT, Apache 2.0) for open-source contributions.