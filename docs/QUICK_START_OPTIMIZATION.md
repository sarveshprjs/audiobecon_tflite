# Quick Start: Optimizing AudioBecon TFLite Performance

This guide provides step-by-step instructions to implement the recommended optimizations for better inference performance.

## Phase 1: Enable Hardware Acceleration (30 minutes)

### Step 1: Update Dependencies

Add the latest TFLite Flutter package to `pubspec.yaml`:

```yaml
dependencies:
  tflite_flutter: ^0.11.0
  tflite_flutter_helper: ^0.3.1
```

### Step 2: Enable GPU Delegate (Android)

Update your TFLite initialization code:

```dart
import 'package:tflite_flutter/tflite_flutter.dart';
import 'dart:io';

Future<Interpreter> createOptimizedInterpreter() async {
  final options = InterpreterOptions();

  if (Platform.isAndroid) {
    try {
      // Try GPU delegate first
      options.addDelegate(GpuDelegateV2(
        options: GpuDelegateOptionsV2(
          isPrecisionLossAllowed: false,
          inferencePreference: TfLiteGpuInferenceUsage.fastSingleAnswer,
        ),
      ));
      print('✅ GPU delegate enabled');
    } catch (e) {
      print('⚠️ GPU delegate failed, trying NNAPI: $e');
      try {
        // Fallback to NNAPI
        options.addDelegate(NnApiDelegate());
        print('✅ NNAPI delegate enabled');
      } catch (e) {
        print('⚠️ NNAPI failed, using CPU: $e');
      }
    }
  } else if (Platform.isIOS) {
    try {
      // Enable Metal delegate for iOS
      options.addDelegate(MetalDelegate());
      print('✅ Metal delegate enabled');
    } catch (e) {
      print('⚠️ Metal delegate failed, using CPU: $e');
    }
  }

  // Set number of threads for CPU fallback
  options.threads = 4;

  return await Interpreter.fromAsset(
    'assets/yamnet.tflite',
    options: options,
  );
}
```

### Step 3: Update Android Configuration

Add to `android/app/build.gradle`:

```gradle
android {
    // ... existing config

    aaptOptions {
        noCompress 'tflite'
    }
}

dependencies {
    // ... existing dependencies

    // GPU delegate support
    implementation 'org.tensorflow:tensorflow-lite-gpu:2.14.0'
}
```

### Step 4: Update iOS Configuration

Add to `ios/Podfile`:

```ruby
post_install do |installer|
  installer.pods_project.targets.each do |target|
    flutter_additional_ios_build_settings(target)

    # Enable Metal for TFLite
    target.build_configurations.each do |config|
      config.build_settings['ENABLE_BITCODE'] = 'NO'
    end
  end
end
```

## Phase 2: Model Quantization (1 hour)

### Step 1: Install TensorFlow

```bash
pip install tensorflow==2.14.0
```

### Step 2: Quantize Your Model

Create `scripts/quantize_model.py`:

```python
import tensorflow as tf
import numpy as np

def representative_dataset():
    """Generate representative data for quantization."""
    for _ in range(100):
        # Generate random audio data (adjust shape to match your model)
        data = np.random.rand(1, 15600).astype(np.float32)
        yield [data]

def quantize_model(input_model_path, output_model_path):
    """Convert model to INT8 quantized version."""
    converter = tf.lite.TFLiteConverter.from_saved_model(input_model_path)

    # Enable INT8 quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    quantized_model = converter.convert()

    with open(output_model_path, 'wb') as f:
        f.write(quantized_model)

    print(f'✅ Quantized model saved to {output_model_path}')

    # Compare sizes
    import os
    original_size = os.path.getsize(input_model_path)
    quantized_size = os.path.getsize(output_model_path)
    reduction = (1 - quantized_size / original_size) * 100

    print(f'Original size: {original_size / 1024:.2f} KB')
    print(f'Quantized size: {quantized_size / 1024:.2f} KB')
    print(f'Size reduction: {reduction:.1f}%')

if __name__ == '__main__':
    quantize_model(
        'models/yamnet_saved_model',
        'assets/yamnet_quantized.tflite'
    )
```

### Step 3: Run Quantization

```bash
python scripts/quantize_model.py
```

### Step 4: Update Asset References

Update `pubspec.yaml`:

```yaml
flutter:
  assets:
    - assets/yamnet.tflite
    - assets/yamnet_quantized.tflite  # Add quantized model
    - assets/yamnet_class_map.csv
```

## Phase 3: Optimize Audio Preprocessing (30 minutes)

### Step 1: Efficient Audio Buffer Management

```dart
class AudioPreprocessor {
  static const int sampleRate = 16000;
  static const int windowSize = 15600; // ~1 second at 16kHz

  final List<double> _buffer = [];

  List<double>? processAudioChunk(List<double> newSamples) {
    _buffer.addAll(newSamples);

    if (_buffer.length >= windowSize) {
      // Extract window
      final window = _buffer.sublist(0, windowSize);

      // Remove processed samples (with overlap)
      final overlap = windowSize ~/ 2;
      _buffer.removeRange(0, windowSize - overlap);

      // Normalize
      return _normalizeAudio(window);
    }

    return null;
  }

  List<double> _normalizeAudio(List<double> samples) {
    // Find max absolute value
    double maxAbs = 0.0;
    for (final sample in samples) {
      final abs = sample.abs();
      if (abs > maxAbs) maxAbs = abs;
    }

    // Normalize to [-1, 1]
    if (maxAbs > 0) {
      return samples.map((s) => s / maxAbs).toList();
    }

    return samples;
  }

  void clear() {
    _buffer.clear();
  }
}
```

### Step 2: Batch Processing

```dart
class BatchInferenceManager {
  final Interpreter _interpreter;
  final int batchSize;

  BatchInferenceManager(this._interpreter, {this.batchSize = 4});

  Future<List<Map<String, dynamic>>> inferBatch(
    List<List<double>> audioChunks,
  ) async {
    final results = <Map<String, dynamic>>[];

    // Process in batches
    for (int i = 0; i < audioChunks.length; i += batchSize) {
      final end = (i + batchSize).clamp(0, audioChunks.length);
      final batch = audioChunks.sublist(i, end);

      // Prepare batch input
      final batchInput = _prepareBatchInput(batch);
      final batchOutput = _prepareBatchOutput(batch.length);

      // Run inference
      _interpreter.run(batchInput, batchOutput);

      // Parse results
      results.addAll(_parseBatchOutput(batchOutput));
    }

    return results;
  }

  List<List<double>> _prepareBatchInput(List<List<double>> batch) {
    return batch;
  }

  List<List<double>> _prepareBatchOutput(int batchSize) {
    return List.generate(batchSize, (_) => List.filled(521, 0.0));
  }

  List<Map<String, dynamic>> _parseBatchOutput(List<List<double>> output) {
    return output.map((scores) {
      final maxScore = scores.reduce((a, b) => a > b ? a : b);
      final maxIndex = scores.indexOf(maxScore);
      return {
        'class_index': maxIndex,
        'confidence': maxScore,
      };
    }).toList();
  }
}
```

## Phase 4: Benchmark Your Improvements (15 minutes)

### Step 1: Create Benchmark Test

```dart
import 'package:flutter_test/flutter_test.dart';
import 'dart:math';

void main() {
  test('Benchmark inference performance', () async {
    final interpreter = await createOptimizedInterpreter();

    // Generate test data
    final random = Random();
    final testData = List.generate(
      15600,
      (_) => random.nextDouble() * 2 - 1,
    );

    // Warmup
    for (int i = 0; i < 10; i++) {
      final input = [testData];
      final output = [List.filled(521, 0.0)];
      interpreter.run(input, output);
    }

    // Benchmark
    final times = <int>[];
    for (int i = 0; i < 100; i++) {
      final stopwatch = Stopwatch()..start();

      final input = [testData];
      final output = [List.filled(521, 0.0)];
      interpreter.run(input, output);

      stopwatch.stop();
      times.add(stopwatch.elapsedMicroseconds);
    }

    // Calculate statistics
    final avgTime = times.reduce((a, b) => a + b) / times.length / 1000;
    final minTime = times.reduce((a, b) => a < b ? a : b) / 1000;
    final maxTime = times.reduce((a, b) => a > b ? a : b) / 1000;

    print('Average inference time: ${avgTime.toStringAsFixed(2)} ms');
    print('Min inference time: ${minTime.toStringAsFixed(2)} ms');
    print('Max inference time: ${maxTime.toStringAsFixed(2)} ms');

    // Assert performance target
    expect(avgTime, lessThan(50.0), reason: 'Inference should be under 50ms');

    interpreter.close();
  });
}
```

### Step 2: Run Benchmark

```bash
flutter test test/benchmark_test.dart
```

## Expected Results

After implementing these optimizations, you should see:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Inference Time (Android GPU) | ~80ms | ~25ms | **3.2x faster** |
| Inference Time (iOS Metal) | ~70ms | ~20ms | **3.5x faster** |
| Model Size | ~5MB | ~1.5MB | **70% smaller** |
| Memory Usage | ~30MB | ~20MB | **33% less** |
| Battery Impact | High | Medium | **~40% better** |

## Troubleshooting

### GPU Delegate Fails on Android

**Problem:** GPU delegate initialization fails

**Solution:**
1. Check device supports OpenGL ES 3.1+
2. Try NNAPI delegate instead
3. Ensure model is compatible with GPU delegate

```dart
// Check GPU compatibility
try {
  final gpuDelegate = GpuDelegateV2();
  print('GPU delegate supported');
} catch (e) {
  print('GPU delegate not supported: $e');
}
```

### Metal Delegate Fails on iOS

**Problem:** Metal delegate not working

**Solution:**
1. Ensure iOS 12.0+
2. Check Metal support in device
3. Verify Podfile configuration

### Quantized Model Accuracy Loss

**Problem:** Quantized model has lower accuracy

**Solution:**
1. Use more representative data in quantization
2. Try dynamic range quantization instead of INT8
3. Use hybrid quantization (weights only)

```python
# Dynamic range quantization (better accuracy)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Don't set representative_dataset
```

## Next Steps

1. ✅ Implement hardware acceleration
2. ✅ Quantize model
3. ✅ Optimize preprocessing
4. ✅ Run benchmarks
5. 📊 Compare with ONNX Runtime (see [IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md))
6. 🚀 Deploy optimized version

## Resources

- [TFLite Performance Best Practices](https://www.tensorflow.org/lite/performance/best_practices)
- [TFLite GPU Delegate Guide](https://www.tensorflow.org/lite/performance/gpu)
- [Model Quantization Guide](https://www.tensorflow.org/lite/performance/post_training_quantization)
- [Audio Classification Tutorial](https://www.tensorflow.org/lite/examples/audio_classification/overview)

---

**Need Help?** Open an issue or check the [full documentation](../README.md).
