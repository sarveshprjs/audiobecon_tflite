import 'dart:io';
import 'package:flutter/foundation.dart';

enum InferenceFramework {
  tfliteCPU,
  tfliteGPU,
  tfliteNNAPI,
  tfliteMetal,
  onnxRuntime,
  coreML,
  pytorchMobile,
}

abstract class InferenceEngine {
  Future<void> initialize();
  Future<Map<String, dynamic>> infer(List<double> audioData);
  Future<void> dispose();
  InferenceFramework get framework;
  bool get isAvailable;
}

class InferenceManager {
  InferenceEngine? _currentEngine;
  InferenceFramework? _currentFramework;

  final Map<InferenceFramework, InferenceEngine> _engines = {};

  InferenceManager();

  Future<void> initialize({InferenceFramework? preferredFramework}) async {
    final framework = preferredFramework ?? _selectBestFramework();

    try {
      _currentEngine = await _createEngine(framework);
      await _currentEngine!.initialize();
      _currentFramework = framework;

      debugPrint('Initialized inference engine: $framework');
    } catch (e) {
      debugPrint('Failed to initialize $framework: $e');

      await _fallbackToAlternative(framework);
    }
  }

  InferenceFramework _selectBestFramework() {
    if (Platform.isIOS) {
      return InferenceFramework.coreML;
    } else if (Platform.isAndroid) {
      return InferenceFramework.tfliteGPU;
    } else if (kIsWeb) {
      return InferenceFramework.tfliteCPU;
    } else {
      return InferenceFramework.onnxRuntime;
    }
  }

  Future<void> _fallbackToAlternative(InferenceFramework failed) async {
    final alternatives = _getAlternativeFrameworks(failed);

    for (final alternative in alternatives) {
      try {
        debugPrint('Trying fallback: $alternative');
        _currentEngine = await _createEngine(alternative);
        await _currentEngine!.initialize();
        _currentFramework = alternative;
        debugPrint('Successfully initialized fallback: $alternative');
        return;
      } catch (e) {
        debugPrint('Fallback $alternative failed: $e');
      }
    }

    throw Exception('All inference frameworks failed to initialize');
  }

  List<InferenceFramework> _getAlternativeFrameworks(InferenceFramework failed) {
    if (Platform.isAndroid) {
      return [
        InferenceFramework.tfliteNNAPI,
        InferenceFramework.tfliteGPU,
        InferenceFramework.tfliteCPU,
        InferenceFramework.onnxRuntime,
      ]..remove(failed);
    } else if (Platform.isIOS) {
      return [
        InferenceFramework.tfliteMetal,
        InferenceFramework.tfliteCPU,
        InferenceFramework.onnxRuntime,
      ]..remove(failed);
    } else {
      return [
        InferenceFramework.onnxRuntime,
        InferenceFramework.tfliteCPU,
      ]..remove(failed);
    }
  }

  Future<InferenceEngine> _createEngine(InferenceFramework framework) async {
    switch (framework) {
      case InferenceFramework.tfliteCPU:
        return TFLiteCPUEngine();
      case InferenceFramework.tfliteGPU:
        return TFLiteGPUEngine();
      case InferenceFramework.tfliteNNAPI:
        return TFLiteNNAPIEngine();
      case InferenceFramework.tfliteMetal:
        return TFLiteMetalEngine();
      case InferenceFramework.onnxRuntime:
        return ONNXRuntimeEngine();
      case InferenceFramework.coreML:
        return CoreMLEngine();
      case InferenceFramework.pytorchMobile:
        return PyTorchMobileEngine();
    }
  }

  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    if (_currentEngine == null) {
      throw Exception('Inference engine not initialized');
    }

    return await _currentEngine!.infer(audioData);
  }

  Future<void> switchFramework(InferenceFramework framework) async {
    if (_currentFramework == framework) {
      return;
    }

    await dispose();
    await initialize(preferredFramework: framework);
  }

  InferenceFramework? get currentFramework => _currentFramework;

  Future<void> dispose() async {
    await _currentEngine?.dispose();
    _currentEngine = null;
    _currentFramework = null;
  }

  Future<Map<InferenceFramework, BenchmarkResult>> benchmarkAll(
    List<double> sampleData,
  ) async {
    final results = <InferenceFramework, BenchmarkResult>{};

    for (final framework in InferenceFramework.values) {
      try {
        final engine = await _createEngine(framework);
        await engine.initialize();

        final result = await _benchmarkEngine(engine, sampleData);
        results[framework] = result;

        await engine.dispose();
      } catch (e) {
        debugPrint('Benchmark failed for $framework: $e');
      }
    }

    return results;
  }

  Future<BenchmarkResult> _benchmarkEngine(
    InferenceEngine engine,
    List<double> sampleData,
  ) async {
    const warmupRuns = 5;
    const benchmarkRuns = 50;

    for (int i = 0; i < warmupRuns; i++) {
      await engine.infer(sampleData);
    }

    final times = <double>[];

    for (int i = 0; i < benchmarkRuns; i++) {
      final stopwatch = Stopwatch()..start();
      await engine.infer(sampleData);
      stopwatch.stop();
      times.add(stopwatch.elapsedMicroseconds / 1000.0);
    }

    final avgTime = times.reduce((a, b) => a + b) / times.length;
    final minTime = times.reduce((a, b) => a < b ? a : b);
    final maxTime = times.reduce((a, b) => a > b ? a : b);

    return BenchmarkResult(
      framework: engine.framework,
      avgInferenceTime: avgTime,
      minInferenceTime: minTime,
      maxInferenceTime: maxTime,
      iterations: benchmarkRuns,
    );
  }
}

class BenchmarkResult {
  final InferenceFramework framework;
  final double avgInferenceTime;
  final double minInferenceTime;
  final double maxInferenceTime;
  final int iterations;

  BenchmarkResult({
    required this.framework,
    required this.avgInferenceTime,
    required this.minInferenceTime,
    required this.maxInferenceTime,
    required this.iterations,
  });

  @override
  String toString() {
    return '''
Framework: $framework
Avg: ${avgInferenceTime.toStringAsFixed(2)}ms
Min: ${minInferenceTime.toStringAsFixed(2)}ms
Max: ${maxInferenceTime.toStringAsFixed(2)}ms
''';
  }
}

class TFLiteCPUEngine implements InferenceEngine {
  @override
  InferenceFramework get framework => InferenceFramework.tfliteCPU;

  @override
  bool get isAvailable => true;

  @override
  Future<void> initialize() async {
    debugPrint('TFLite CPU engine initialized');
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    await Future.delayed(const Duration(milliseconds: 50));
    return {'result': 'tflite_cpu'};
  }

  @override
  Future<void> dispose() async {
    debugPrint('TFLite CPU engine disposed');
  }
}

class TFLiteGPUEngine implements InferenceEngine {
  @override
  InferenceFramework get framework => InferenceFramework.tfliteGPU;

  @override
  bool get isAvailable => Platform.isAndroid;

  @override
  Future<void> initialize() async {
    if (!isAvailable) {
      throw Exception('GPU delegate not available on this platform');
    }
    debugPrint('TFLite GPU engine initialized');
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    await Future.delayed(const Duration(milliseconds: 20));
    return {'result': 'tflite_gpu'};
  }

  @override
  Future<void> dispose() async {
    debugPrint('TFLite GPU engine disposed');
  }
}

class TFLiteNNAPIEngine implements InferenceEngine {
  @override
  InferenceFramework get framework => InferenceFramework.tfliteNNAPI;

  @override
  bool get isAvailable => Platform.isAndroid;

  @override
  Future<void> initialize() async {
    if (!isAvailable) {
      throw Exception('NNAPI delegate not available on this platform');
    }
    debugPrint('TFLite NNAPI engine initialized');
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    await Future.delayed(const Duration(milliseconds: 25));
    return {'result': 'tflite_nnapi'};
  }

  @override
  Future<void> dispose() async {
    debugPrint('TFLite NNAPI engine disposed');
  }
}

class TFLiteMetalEngine implements InferenceEngine {
  @override
  InferenceFramework get framework => InferenceFramework.tfliteMetal;

  @override
  bool get isAvailable => Platform.isIOS;

  @override
  Future<void> initialize() async {
    if (!isAvailable) {
      throw Exception('Metal delegate not available on this platform');
    }
    debugPrint('TFLite Metal engine initialized');
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    await Future.delayed(const Duration(milliseconds: 15));
    return {'result': 'tflite_metal'};
  }

  @override
  Future<void> dispose() async {
    debugPrint('TFLite Metal engine disposed');
  }
}

class ONNXRuntimeEngine implements InferenceEngine {
  @override
  InferenceFramework get framework => InferenceFramework.onnxRuntime;

  @override
  bool get isAvailable => true;

  @override
  Future<void> initialize() async {
    debugPrint('ONNX Runtime engine initialized');
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    await Future.delayed(const Duration(milliseconds: 30));
    return {'result': 'onnx_runtime'};
  }

  @override
  Future<void> dispose() async {
    debugPrint('ONNX Runtime engine disposed');
  }
}

class CoreMLEngine implements InferenceEngine {
  @override
  InferenceFramework get framework => InferenceFramework.coreML;

  @override
  bool get isAvailable => Platform.isIOS;

  @override
  Future<void> initialize() async {
    if (!isAvailable) {
      throw Exception('Core ML not available on this platform');
    }
    debugPrint('Core ML engine initialized');
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    await Future.delayed(const Duration(milliseconds: 10));
    return {'result': 'coreml'};
  }

  @override
  Future<void> dispose() async {
    debugPrint('Core ML engine disposed');
  }
}

class PyTorchMobileEngine implements InferenceEngine {
  @override
  InferenceFramework get framework => InferenceFramework.pytorchMobile;

  @override
  bool get isAvailable => Platform.isAndroid || Platform.isIOS;

  @override
  Future<void> initialize() async {
    if (!isAvailable) {
      throw Exception('PyTorch Mobile not available on this platform');
    }
    debugPrint('PyTorch Mobile engine initialized');
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    await Future.delayed(const Duration(milliseconds: 35));
    return {'result': 'pytorch_mobile'};
  }

  @override
  Future<void> dispose() async {
    debugPrint('PyTorch Mobile engine disposed');
  }
}
