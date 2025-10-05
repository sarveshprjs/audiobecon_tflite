import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'inference_benchmark.dart';

class TFLiteEngine implements InferenceEngine {
  Interpreter? _interpreter;

  @override
  String get name => 'TensorFlow Lite';

  @override
  Future<void> initialize() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/yamnet.tflite');
      debugPrint('TFLite interpreter initialized');
    } catch (e) {
      debugPrint('Error initializing TFLite: $e');
      rethrow;
    }
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    if (_interpreter == null) {
      throw Exception('Interpreter not initialized');
    }

    try {
      final input = audioData.reshape([1, audioData.length]);
      final output = List.filled(1 * 521, 0.0).reshape([1, 521]);

      _interpreter!.run(input, output);

      final scores = output[0] as List<double>;
      final maxScore = scores.reduce((a, b) => a > b ? a : b);
      final maxIndex = scores.indexOf(maxScore);

      return {
        'class_index': maxIndex,
        'confidence': maxScore,
        'all_scores': scores,
      };
    } catch (e) {
      debugPrint('TFLite inference error: $e');
      rethrow;
    }
  }

  @override
  Future<void> dispose() async {
    _interpreter?.close();
    _interpreter = null;
  }
}

class TFLiteGPUEngine implements InferenceEngine {
  Interpreter? _interpreter;

  @override
  String get name => 'TensorFlow Lite (GPU Delegate)';

  @override
  Future<void> initialize() async {
    try {
      final options = InterpreterOptions()
        ..addDelegate(GpuDelegateV2());

      _interpreter = await Interpreter.fromAsset(
        'assets/yamnet.tflite',
        options: options,
      );
      debugPrint('TFLite GPU interpreter initialized');
    } catch (e) {
      debugPrint('Error initializing TFLite GPU: $e');
      rethrow;
    }
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    if (_interpreter == null) {
      throw Exception('Interpreter not initialized');
    }

    try {
      final input = audioData.reshape([1, audioData.length]);
      final output = List.filled(1 * 521, 0.0).reshape([1, 521]);

      _interpreter!.run(input, output);

      final scores = output[0] as List<double>;
      final maxScore = scores.reduce((a, b) => a > b ? a : b);
      final maxIndex = scores.indexOf(maxScore);

      return {
        'class_index': maxIndex,
        'confidence': maxScore,
        'all_scores': scores,
      };
    } catch (e) {
      debugPrint('TFLite GPU inference error: $e');
      rethrow;
    }
  }

  @override
  Future<void> dispose() async {
    _interpreter?.close();
    _interpreter = null;
  }
}

class TFLiteNNAPIEngine implements InferenceEngine {
  Interpreter? _interpreter;

  @override
  String get name => 'TensorFlow Lite (NNAPI Delegate)';

  @override
  Future<void> initialize() async {
    try {
      final options = InterpreterOptions()
        ..addDelegate(NnApiDelegate());

      _interpreter = await Interpreter.fromAsset(
        'assets/yamnet.tflite',
        options: options,
      );
      debugPrint('TFLite NNAPI interpreter initialized');
    } catch (e) {
      debugPrint('Error initializing TFLite NNAPI: $e');
      rethrow;
    }
  }

  @override
  Future<Map<String, dynamic>> infer(List<double> audioData) async {
    if (_interpreter == null) {
      throw Exception('Interpreter not initialized');
    }

    try {
      final input = audioData.reshape([1, audioData.length]);
      final output = List.filled(1 * 521, 0.0).reshape([1, 521]);

      _interpreter!.run(input, output);

      final scores = output[0] as List<double>;
      final maxScore = scores.reduce((a, b) => a > b ? a : b);
      final maxIndex = scores.indexOf(maxScore);

      return {
        'class_index': maxIndex,
        'confidence': maxScore,
        'all_scores': scores,
      };
    } catch (e) {
      debugPrint('TFLite NNAPI inference error: $e');
      rethrow;
    }
  }

  @override
  Future<void> dispose() async {
    _interpreter?.close();
    _interpreter = null;
  }
}
