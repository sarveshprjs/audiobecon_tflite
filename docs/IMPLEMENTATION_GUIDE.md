# Implementation Guide: Hybrid Inference Approach

This guide provides step-by-step instructions for implementing a hybrid inference system that supports multiple ML frameworks while maintaining TensorFlow Lite as the default option.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              Flutter Application Layer               │
└─────────────────────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│          Inference Engine Abstraction Layer         │
│  (InferenceEngine interface + Factory Pattern)      │
└─────────────────────────────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
    ┌────────┐     ┌──────────┐    ┌──────────┐
    │ TFLite │     │   ONNX   │    │ CoreML   │
    │ Engine │     │  Runtime │    │  Engine  │
    └────────┘     └──────────┘    └──────────┘
```

---

## Phase 1: Create Abstraction Layer

### Step 1.1: Define Inference Engine Interface

Create `lib/inference/inference_engine.dart`:

```dart
abstract class InferenceEngine {
  Future<void> initialize(String modelPath);
  Future<Map<String, dynamic>?> classify(List<double> audioData);
  Future<void> dispose();
  String get engineName;
  bool get isHardwareAccelerated;
}
```

### Step 1.2: Create TFLite Implementation

Create `lib/inference/tflite_engine.dart`:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'inference_engine.dart';

class TFLiteEngine implements InferenceEngine {
  Interpreter? _interpreter;

  @override
  String get engineName => 'TensorFlow Lite';

  @override
  bool get isHardwareAccelerated => _interpreter?.isAllocated ?? false;

  @override
  Future<void> initialize(String modelPath) async {
    final options = InterpreterOptions()
      ..threads = 4
      ..useNnApiForAndroid = true;

    _interpreter = await Interpreter.fromAsset(
      modelPath,
      options: options,
    );
  }

  @override
  Future<Map<String, dynamic>?> classify(List<double> audioData) async {
    if (_interpreter == null) return null;

    // Prepare input tensor
    final input = audioData.reshape([1, audioData.length]);
    final output = List.filled(1 * 521, 0.0).reshape([1, 521]);

    // Run inference
    _interpreter!.run(input, output);

    // Process output
    final scores = output[0] as List<double>;
    final maxIndex = scores.indexOf(scores.reduce((a, b) => a > b ? a : b));

    return {
      'label': _getLabel(maxIndex),
      'confidence': scores[maxIndex],
      'scores': scores,
    };
  }

  @override
  Future<void> dispose() async {
    _interpreter?.close();
    _interpreter = null;
  }

  String _getLabel(int index) {
    // Load from yamnet_class_map.csv
    return 'Label_$index';
  }
}
```

### Step 1.3: Create ONNX Runtime Implementation

Create `lib/inference/onnx_engine.dart`:

```dart
import 'package:onnxruntime/onnxruntime.dart';
import 'inference_engine.dart';

class ONNXEngine implements InferenceEngine {
  OrtSession? _session;

  @override
  String get engineName => 'ONNX Runtime';

  @override
  bool get isHardwareAccelerated => true;

  @override
  Future<void> initialize(String modelPath) async {
    final sessionOptions = OrtSessionOptions()
      ..setInterOpNumThreads(4)
      ..setIntraOpNumThreads(4)
      ..setSessionGraphOptimizationLevel(
        GraphOptimizationLevel.ortEnableAll
      );

    _session = OrtSession.fromFile(modelPath, sessionOptions);
  }

  @override
  Future<Map<String, dynamic>?> classify(List<double> audioData) async {
    if (_session == null) return null;

    final inputOrt = OrtValueTensor.createTensorWithDataList(
      audioData,
      [1, audioData.length],
    );

    final outputs = _session!.run([inputOrt]);
    final scores = outputs[0].value as List<double>;

    final maxIndex = scores.indexOf(scores.reduce((a, b) => a > b ? a : b));

    return {
      'label': _getLabel(maxIndex),
      'confidence': scores[maxIndex],
      'scores': scores,
    };
  }

  @override
  Future<void> dispose() async {
    _session?.release();
    _session = null;
  }

  String _getLabel(int index) {
    return 'Label_$index';
  }
}
```

### Step 1.4: Create CoreML Implementation (iOS)

Create `lib/inference/coreml_engine.dart`:

```dart
import 'dart:io';
import 'package:flutter/services.dart';
import 'inference_engine.dart';

class CoreMLEngine implements InferenceEngine {
  static const platform = MethodChannel('com.audiobecon/coreml');
  bool _initialized = false;

  @override
  String get engineName => 'Core ML';

  @override
  bool get isHardwareAccelerated => true;

  @override
  Future<void> initialize(String modelPath) async {
    if (!Platform.isIOS) {
      throw UnsupportedError('CoreML is only available on iOS');
    }

    await platform.invokeMethod('initializeModel', {'path': modelPath});
    _initialized = true;
  }

  @override
  Future<Map<String, dynamic>?> classify(List<double> audioData) async {
    if (!_initialized) return null;

    final result = await platform.invokeMethod('classify', {
      'audioData': audioData,
    });

    return Map<String, dynamic>.from(result);
  }

  @override
  Future<void> dispose() async {
    if (_initialized) {
      await platform.invokeMethod('dispose');
      _initialized = false;
    }
  }
}
```

---

## Phase 2: Implement Factory Pattern

Create `lib/inference/inference_factory.dart`:

```dart
import 'dart:io';
import 'inference_engine.dart';
import 'tflite_engine.dart';
import 'onnx_engine.dart';
import 'coreml_engine.dart';

enum InferenceBackend {
  tflite,
  onnx,
  coreml,
  auto,
}

class InferenceFactory {
  static InferenceEngine create(InferenceBackend backend) {
    switch (backend) {
      case InferenceBackend.tflite:
        return TFLiteEngine();

      case InferenceBackend.onnx:
        return ONNXEngine();

      case InferenceBackend.coreml:
        if (!Platform.isIOS) {
          throw UnsupportedError('CoreML is only available on iOS');
        }
        return CoreMLEngine();

      case InferenceBackend.auto:
        return _selectBestEngine();
    }
  }

  static InferenceEngine _selectBestEngine() {
    if (Platform.isIOS) {
      return CoreMLEngine();
    } else if (Platform.isAndroid) {
      return TFLiteEngine();
    } else {
      return ONNXEngine();
    }
  }

  static List<InferenceBackend> getAvailableBackends() {
    final backends = <InferenceBackend>[InferenceBackend.tflite];

    if (Platform.isIOS) {
      backends.add(InferenceBackend.coreml);
    }

    backends.add(InferenceBackend.onnx);
    backends.add(InferenceBackend.auto);

    return backends;
  }
}
```

---

## Phase 3: Update Main Application

Modify `lib/main.dart` to use the abstraction layer:

```dart
import 'inference/inference_engine.dart';
import 'inference/inference_factory.dart';

class SoundClassifier {
  late InferenceEngine _engine;

  Future<void> loadModel({InferenceBackend backend = InferenceBackend.auto}) async {
    _engine = InferenceFactory.create(backend);
    await _engine.initialize('assets/yamnet.tflite');
    debugPrint('Loaded ${_engine.engineName}');
  }

  Future<Map<String, dynamic>?> classifySound(List<double> audioData) async {
    return await _engine.classify(audioData);
  }

  void dispose() {
    _engine.dispose();
  }
}
```

---

## Phase 4: Add Configuration UI

Add settings page to allow users to select inference backend:

```dart
class SettingsPage extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Settings')),
      body: ListView(
        children: [
          ListTile(
            title: Text('Inference Backend'),
            subtitle: Text('Select ML framework'),
            trailing: DropdownButton<InferenceBackend>(
              items: InferenceFactory.getAvailableBackends()
                .map((backend) => DropdownMenuItem(
                  value: backend,
                  child: Text(backend.toString()),
                ))
                .toList(),
              onChanged: (value) {
                // Update backend
              },
            ),
          ),
        ],
      ),
    );
  }
}
```

---

## Phase 5: Testing & Benchmarking

1. **Unit Tests**: Test each engine implementation
2. **Integration Tests**: Test factory pattern and engine switching
3. **Performance Tests**: Benchmark inference speed on target devices
4. **Memory Tests**: Monitor memory usage across engines

---

## Dependencies to Add

Update `pubspec.yaml`:

```yaml
dependencies:
  tflite_flutter: ^0.11.0
  onnxruntime: ^1.16.0  # For ONNX support
  # CoreML uses native iOS APIs via MethodChannel
```

---

## Next Steps

1. Implement model conversion scripts (TFLite → ONNX, TFLite → CoreML)
2. Add telemetry to track performance metrics
3. Create automated benchmarking suite
4. Document platform-specific optimizations

---

## Conclusion

This hybrid approach provides:
- **Flexibility**: Easy to add new inference engines
- **Performance**: Use best engine per platform
- **Maintainability**: Clean abstraction layer
- **Backward Compatibility**: TFLite remains default
