import 'package:flutter_test/flutter_test.dart';
import 'package:sound_hazard_app/inference/inference_manager.dart';

void main() {
  group('InferenceManager', () {
    late InferenceManager manager;

    setUp(() {
      manager = InferenceManager();
    });

    tearDown(() async {
      await manager.dispose();
    });

    test('initializes with default framework', () async {
      await manager.initialize();
      expect(manager.currentFramework, isNotNull);
    });

    test('performs inference successfully', () async {
      await manager.initialize(
        preferredFramework: InferenceFramework.tfliteCPU,
      );

      final sampleData = List.generate(1000, (i) => i.toDouble());
      final result = await manager.infer(sampleData);

      expect(result, isNotNull);
      expect(result, isA<Map<String, dynamic>>());
    });

    test('switches between frameworks', () async {
      await manager.initialize(
        preferredFramework: InferenceFramework.tfliteCPU,
      );
      expect(manager.currentFramework, InferenceFramework.tfliteCPU);

      await manager.switchFramework(InferenceFramework.onnxRuntime);
      expect(manager.currentFramework, InferenceFramework.onnxRuntime);
    });

    test('handles inference without initialization', () async {
      final sampleData = List.generate(1000, (i) => i.toDouble());

      expect(
        () => manager.infer(sampleData),
        throwsException,
      );
    });

    test('disposes resources properly', () async {
      await manager.initialize();
      await manager.dispose();

      expect(manager.currentFramework, isNull);
    });
  });

  group('BenchmarkResult', () {
    test('creates valid benchmark result', () {
      final result = BenchmarkResult(
        framework: InferenceFramework.tfliteCPU,
        avgInferenceTime: 50.0,
        minInferenceTime: 40.0,
        maxInferenceTime: 60.0,
        iterations: 100,
      );

      expect(result.framework, InferenceFramework.tfliteCPU);
      expect(result.avgInferenceTime, 50.0);
      expect(result.minInferenceTime, 40.0);
      expect(result.maxInferenceTime, 60.0);
      expect(result.iterations, 100);
    });

    test('formats benchmark result as string', () {
      final result = BenchmarkResult(
        framework: InferenceFramework.tfliteGPU,
        avgInferenceTime: 25.5,
        minInferenceTime: 20.0,
        maxInferenceTime: 30.0,
        iterations: 50,
      );

      final str = result.toString();
      expect(str, contains('tfliteGPU'));
      expect(str, contains('25.50ms'));
    });
  });

  group('InferenceEngine implementations', () {
    test('TFLiteCPUEngine is always available', () {
      final engine = TFLiteCPUEngine();
      expect(engine.isAvailable, isTrue);
      expect(engine.framework, InferenceFramework.tfliteCPU);
    });

    test('ONNXRuntimeEngine is always available', () {
      final engine = ONNXRuntimeEngine();
      expect(engine.isAvailable, isTrue);
      expect(engine.framework, InferenceFramework.onnxRuntime);
    });

    test('engine initializes and performs inference', () async {
      final engine = TFLiteCPUEngine();
      await engine.initialize();

      final sampleData = List.generate(1000, (i) => i.toDouble());
      final result = await engine.infer(sampleData);

      expect(result, isNotNull);
      expect(result, isA<Map<String, dynamic>>());

      await engine.dispose();
    });
  });
}
