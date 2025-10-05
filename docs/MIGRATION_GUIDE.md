# Migration Guide: Switching Inference Frameworks

This guide helps you migrate from one inference framework to another in the AudioBecon TFLite project.

## Table of Contents
- [TFLite to ONNX Runtime](#tflite-to-onnx-runtime)
- [TFLite to Core ML (iOS)](#tflite-to-core-ml-ios)
- [TFLite to PyTorch Mobile](#tflite-to-pytorch-mobile)
- [Hybrid Implementation](#hybrid-implementation)

---

## TFLite to ONNX Runtime

### Why Migrate?
- 20-40% faster inference on many platforms
- Better CPU optimization
- Framework-agnostic model support

### Prerequisites
```bash
pip install tf2onnx onnx onnxruntime
```

### Step 1: Convert Model

```python
# convert_to_onnx.py
import tensorflow as tf
import tf2onnx
import onnx

def convert_tflite_to_onnx(tflite_path, onnx_path):
    """Convert TFLite model to ONNX format."""

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input/output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")

    # Convert to ONNX
    # Note: Direct TFLite to ONNX conversion is complex
    # Better approach: Convert from original TensorFlow model

    # If you have the original TF model:
    model = tf.saved_model.load('path/to/saved_model')

    spec = (tf.TensorSpec(input_details[0]['shape'], tf.float32, name="input"),)
    output_path = onnx_path

    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=output_path
    )

    print(f"✅ ONNX model saved to {output_path}")

    # Verify model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("✅ ONNX model is valid")

if __name__ == '__main__':
    convert_tflite_to_onnx(
        'assets/yamnet.tflite',
        'assets/yamnet.onnx'
    )
```

### Step 2: Add ONNX Runtime Dependency

Update `pubspec.yaml`:
```yaml
dependencies:
  onnxruntime: ^1.16.0  # Check for latest version
```

### Step 3: Implement ONNX Engine

```dart
import 'package:onnxruntime/onnxruntime.dart';

class ONNXInferenceEngine {
  OrtSession? _session;
  OrtSessionOptions? _sessionOptions;

  Future<void> initialize() async {
    try {
      // Initialize ONNX Runtime
      OrtEnv.instance.init();

      // Create session options
      _sessionOptions = OrtSessionOptions()
        ..setInterOpNumThreads(4)
        ..setIntraOpNumThreads(4)
        ..setSessionGraphOptimizationLevel(
          GraphOptimizationLevel.ortEnableAll,
        );

      // Load model
      final modelPath = await _getModelPath('assets/yamnet.onnx');
      _session = OrtSession.fromFile(modelPath, _sessionOptions!);

      print('✅ ONNX Runtime initialized');
    } catch (e) {
      print('❌ ONNX initialization failed: $e');
      rethrow;
    }
  }

  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    if (_session == null) {
      throw Exception('Session not initialized');
    }

    try {
      // Prepare input
      final inputShape = [1, audioData.length];
      final inputTensor = OrtValueTensor.createTensorWithDataList(
        audioData,
        inputShape,
      );

      // Run inference
      final inputs = {'input': inputTensor};
      final outputs = await _session!.runAsync(
        OrtRunOptions(),
        inputs,
      );

      // Parse output
      final outputTensor = outputs[0];
      final scores = outputTensor?.value as List<double>;

      final maxScore = scores.reduce((a, b) => a > b ? a : b);
      final maxIndex = scores.indexOf(maxScore);

      // Release tensors
      inputTensor.release();
      outputTensor?.release();

      return {
        'class_index': maxIndex,
        'confidence': maxScore,
        'all_scores': scores,
      };
    } catch (e) {
      print('❌ ONNX inference failed: $e');
      rethrow;
    }
  }

  Future<void> dispose() async {
    _session?.release();
    _sessionOptions?.release();
    _session = null;
    _sessionOptions = null;
  }

  Future<String> _getModelPath(String assetPath) async {
    // Copy asset to temporary directory
    final bytes = await rootBundle.load(assetPath);
    final tempDir = await getTemporaryDirectory();
    final file = File('${tempDir.path}/yamnet.onnx');
    await file.writeAsBytes(bytes.buffer.asUint8List());
    return file.path;
  }
}
```

### Step 4: Update Main App

```dart
// Replace TFLite initialization with ONNX
final inferenceEngine = ONNXInferenceEngine();
await inferenceEngine.initialize();

// Use same interface
final result = await inferenceEngine.infer(audioData);
```

### Step 5: Benchmark

```dart
// Compare performance
final tfliteTime = await benchmarkTFLite(audioData);
final onnxTime = await benchmarkONNX(audioData);

print('TFLite: ${tfliteTime}ms');
print('ONNX: ${onnxTime}ms');
print('Speedup: ${(tfliteTime / onnxTime).toStringAsFixed(2)}x');
```

---

## TFLite to Core ML (iOS)

### Why Migrate?
- 50-100% faster on iOS devices
- Better battery efficiency
- Native Apple integration

### Prerequisites
```bash
pip install coremltools
```

### Step 1: Convert Model

```python
# convert_to_coreml.py
import coremltools as ct
import tensorflow as tf

def convert_tflite_to_coreml(tflite_path, coreml_path):
    """Convert TFLite model to Core ML format."""

    # Load TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input details
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']

    # Better: Convert from original TensorFlow model
    # Load saved model
    tf_model = tf.saved_model.load('path/to/saved_model')

    # Convert to Core ML
    mlmodel = ct.convert(
        tf_model,
        inputs=[ct.TensorType(shape=input_shape, name="audio_input")],
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=ct.precision.FLOAT16,  # Use FP16 for better performance
    )

    # Set metadata
    mlmodel.author = 'AudioBecon Team'
    mlmodel.short_description = 'Audio classification model'
    mlmodel.version = '1.0'

    # Save
    mlmodel.save(coreml_path)
    print(f'✅ Core ML model saved to {coreml_path}')

if __name__ == '__main__':
    convert_tflite_to_coreml(
        'assets/yamnet.tflite',
        'assets/yamnet.mlmodel'
    )
```

### Step 2: Add Core ML to iOS Project

1. Add model to Xcode project:
   - Drag `yamnet.mlmodel` into Xcode
   - Ensure "Target Membership" is checked

2. Create Swift bridge:

```swift
// CoreMLBridge.swift
import Foundation
import CoreML

@objc class CoreMLBridge: NSObject {
    private var model: yamnet?

    @objc func initialize() -> Bool {
        do {
            let config = MLModelConfiguration()
            config.computeUnits = .all  // Use Neural Engine if available
            model = try yamnet(configuration: config)
            return true
        } catch {
            print("Failed to load Core ML model: \(error)")
            return false
        }
    }

    @objc func predict(_ audioData: [Double]) -> [String: Any]? {
        guard let model = model else { return nil }

        do {
            // Prepare input
            let input = try MLMultiArray(shape: [1, NSNumber(value: audioData.count)], dataType: .double)
            for (index, value) in audioData.enumerated() {
                input[index] = NSNumber(value: value)
            }

            // Run prediction
            let output = try model.prediction(audio_input: input)

            // Parse output
            let scores = output.scores
            var maxScore = 0.0
            var maxIndex = 0

            for i in 0..<scores.count {
                let score = scores[i].doubleValue
                if score > maxScore {
                    maxScore = score
                    maxIndex = i
                }
            }

            return [
                "class_index": maxIndex,
                "confidence": maxScore
            ]
        } catch {
            print("Prediction failed: \(error)")
            return nil
        }
    }
}
```

### Step 3: Create Flutter Plugin

```dart
// lib/platform/coreml_plugin.dart
import 'package:flutter/services.dart';

class CoreMLPlugin {
  static const MethodChannel _channel = MethodChannel('coreml_plugin');

  Future<bool> initialize() async {
    try {
      final result = await _channel.invokeMethod('initialize');
      return result as bool;
    } catch (e) {
      print('CoreML initialization failed: $e');
      return false;
    }
  }

  Future<Map<String, dynamic>?> predict(List<double> audioData) async {
    try {
      final result = await _channel.invokeMethod('predict', {
        'audioData': audioData,
      });
      return Map<String, dynamic>.from(result);
    } catch (e) {
      print('CoreML prediction failed: $e');
      return null;
    }
  }
}
```

### Step 4: Register Plugin

```swift
// ios/Runner/AppDelegate.swift
import UIKit
import Flutter

@UIApplicationMain
@objc class AppDelegate: FlutterAppDelegate {
    private let coreMLBridge = CoreMLBridge()

    override func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        let controller = window?.rootViewController as! FlutterViewController
        let channel = FlutterMethodChannel(
            name: "coreml_plugin",
            binaryMessenger: controller.binaryMessenger
        )

        channel.setMethodCallHandler { [weak self] (call, result) in
            guard let self = self else { return }

            switch call.method {
            case "initialize":
                let success = self.coreMLBridge.initialize()
                result(success)
            case "predict":
                if let args = call.arguments as? [String: Any],
                   let audioData = args["audioData"] as? [Double] {
                    let prediction = self.coreMLBridge.predict(audioData)
                    result(prediction)
                } else {
                    result(FlutterError(code: "INVALID_ARGS", message: nil, details: nil))
                }
            default:
                result(FlutterMethodNotImplemented)
            }
        }

        GeneratedPluginRegistrant.register(with: self)
        return super.application(application, didFinishLaunchingWithOptions: launchOptions)
    }
}
```

---

## Hybrid Implementation

### Best of Both Worlds

Implement a unified interface that automatically selects the best framework:

```dart
// lib/inference/unified_inference.dart
import 'dart:io';
import 'package:flutter/foundation.dart';

class UnifiedInferenceEngine {
  late final InferenceBackend _backend;

  Future<void> initialize() async {
    if (Platform.isIOS) {
      // Try Core ML first on iOS
      try {
        _backend = CoreMLBackend();
        await _backend.initialize();
        print('✅ Using Core ML');
        return;
      } catch (e) {
        print('⚠️ Core ML failed, falling back to TFLite');
      }
    }

    if (Platform.isAndroid || Platform.isIOS) {
      // Try TFLite with hardware acceleration
      try {
        _backend = TFLiteBackend(useGPU: true);
        await _backend.initialize();
        print('✅ Using TFLite with GPU');
        return;
      } catch (e) {
        print('⚠️ GPU failed, trying NNAPI');
      }

      try {
        _backend = TFLiteBackend(useNNAPI: true);
        await _backend.initialize();
        print('✅ Using TFLite with NNAPI');
        return;
      } catch (e) {
        print('⚠️ NNAPI failed, using CPU');
      }
    }

    // Fallback to CPU
    _backend = TFLiteBackend();
    await _backend.initialize();
    print('✅ Using TFLite CPU');
  }

  Future<Map<String, dynamic>> infer(List<double> audioData) {
    return _backend.infer(audioData);
  }

  Future<void> dispose() {
    return _backend.dispose();
  }

  String get backendName => _backend.name;
}

abstract class InferenceBackend {
  Future<void> initialize();
  Future<Map<String, dynamic>> infer(List<double> audioData);
  Future<void> dispose();
  String get name;
}
```

---

## Performance Comparison

After migration, benchmark your results:

```dart
Future<void> compareFrameworks() async {
  final sampleData = List.generate(15600, (i) => i.toDouble());

  // Test TFLite
  final tflite = TFLiteBackend();
  await tflite.initialize();
  final tfliteTime = await benchmark(tflite, sampleData);
  await tflite.dispose();

  // Test ONNX
  final onnx = ONNXBackend();
  await onnx.initialize();
  final onnxTime = await benchmark(onnx, sampleData);
  await onnx.dispose();

  // Test Core ML (iOS only)
  if (Platform.isIOS) {
    final coreml = CoreMLBackend();
    await coreml.initialize();
    final coremlTime = await benchmark(coreml, sampleData);
    await coreml.dispose();

    print('Core ML: ${coremlTime}ms');
  }

  print('TFLite: ${tfliteTime}ms');
  print('ONNX: ${onnxTime}ms');
}

Future<double> benchmark(InferenceBackend backend, List<double> data) async {
  final times = <int>[];

  // Warmup
  for (int i = 0; i < 10; i++) {
    await backend.infer(data);
  }

  // Benchmark
  for (int i = 0; i < 100; i++) {
    final stopwatch = Stopwatch()..start();
    await backend.infer(data);
    stopwatch.stop();
    times.add(stopwatch.elapsedMicroseconds);
  }

  return times.reduce((a, b) => a + b) / times.length / 1000;
}
```

---

## Rollback Plan

If migration doesn't work as expected:

1. **Keep old code**: Don't delete TFLite implementation
2. **Feature flag**: Use runtime flag to switch frameworks
3. **A/B testing**: Test with subset of users first
4. **Monitoring**: Track performance metrics
5. **Quick rollback**: Revert to TFLite if issues arise

```dart
class FeatureFlags {
  static bool useONNX = false;  // Set to true to enable ONNX
  static bool useCoreML = false;  // Set to true to enable Core ML
}

Future<void> initializeInference() async {
  if (Platform.isIOS && FeatureFlags.useCoreML) {
    return initializeCoreML();
  } else if (FeatureFlags.useONNX) {
    return initializeONNX();
  } else {
    return initializeTFLite();
  }
}
```

---

## Troubleshooting

### Common Issues

**Issue: Model conversion fails**
- Ensure you have the original TensorFlow model
- Check model compatibility with target framework
- Try different conversion tools

**Issue: Performance worse than TFLite**
- Verify hardware acceleration is enabled
- Check model quantization settings
- Ensure proper input preprocessing

**Issue: Framework not available on device**
- Implement proper fallback logic
- Test on multiple devices
- Check minimum OS requirements

---

## Conclusion

Migration between frameworks is straightforward with proper planning:

1. ✅ Convert model to target format
2. ✅ Implement new inference engine
3. ✅ Benchmark performance
4. ✅ Implement fallback logic
5. ✅ Test thoroughly
6. ✅ Monitor in production

**Recommendation:** Start with TFLite optimization before migrating to ensure you actually need the alternative framework.

---

**Need Help?** Check the [full documentation](../README.md) or open an issue.
