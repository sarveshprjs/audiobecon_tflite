import 'dart:async';
import 'package:flutter/foundation.dart';

class BenchmarkResult {
  final String framework;
  final double avgInferenceTime;
  final double minInferenceTime;
  final double maxInferenceTime;
  final double memoryUsage;
  final int iterations;
  final DateTime timestamp;

  BenchmarkResult({
    required this.framework,
    required this.avgInferenceTime,
    required this.minInferenceTime,
    required this.maxInferenceTime,
    required this.memoryUsage,
    required this.iterations,
    required this.timestamp,
  });

  Map<String, dynamic> toJson() => {
        'framework': framework,
        'avgInferenceTime': avgInferenceTime,
        'minInferenceTime': minInferenceTime,
        'maxInferenceTime': maxInferenceTime,
        'memoryUsage': memoryUsage,
        'iterations': iterations,
        'timestamp': timestamp.toIso8601String(),
      };

  @override
  String toString() {
    return '''
Framework: $framework
Average Inference Time: ${avgInferenceTime.toStringAsFixed(2)} ms
Min Inference Time: ${minInferenceTime.toStringAsFixed(2)} ms
Max Inference Time: ${maxInferenceTime.toStringAsFixed(2)} ms
Memory Usage: ${memoryUsage.toStringAsFixed(2)} MB
Iterations: $iterations
Timestamp: $timestamp
''';
  }
}

abstract class InferenceEngine {
  Future<void> initialize();
  Future<Map<String, dynamic>> infer(List<double> audioData);
  Future<void> dispose();
  String get name;
}

class InferenceBenchmark {
  final List<InferenceEngine> engines;
  final int warmupIterations;
  final int benchmarkIterations;

  InferenceBenchmark({
    required this.engines,
    this.warmupIterations = 10,
    this.benchmarkIterations = 100,
  });

  Future<List<BenchmarkResult>> runBenchmarks(List<double> sampleAudioData) async {
    final results = <BenchmarkResult>[];

    for (final engine in engines) {
      debugPrint('Benchmarking ${engine.name}...');

      try {
        await engine.initialize();

        for (int i = 0; i < warmupIterations; i++) {
          await engine.infer(sampleAudioData);
        }

        final inferenceTimes = <double>[];
        final memoryBefore = _getMemoryUsage();

        for (int i = 0; i < benchmarkIterations; i++) {
          final stopwatch = Stopwatch()..start();
          await engine.infer(sampleAudioData);
          stopwatch.stop();
          inferenceTimes.add(stopwatch.elapsedMicroseconds / 1000.0);
        }

        final memoryAfter = _getMemoryUsage();
        final memoryUsed = memoryAfter - memoryBefore;

        final avgTime = inferenceTimes.reduce((a, b) => a + b) / inferenceTimes.length;
        final minTime = inferenceTimes.reduce((a, b) => a < b ? a : b);
        final maxTime = inferenceTimes.reduce((a, b) => a > b ? a : b);

        results.add(BenchmarkResult(
          framework: engine.name,
          avgInferenceTime: avgTime,
          minInferenceTime: minTime,
          maxInferenceTime: maxTime,
          memoryUsage: memoryUsed,
          iterations: benchmarkIterations,
          timestamp: DateTime.now(),
        ));

        await engine.dispose();

        debugPrint('${engine.name} benchmark completed');
      } catch (e) {
        debugPrint('Error benchmarking ${engine.name}: $e');
      }
    }

    return results;
  }

  double _getMemoryUsage() {
    return 0.0;
  }

  static String generateReport(List<BenchmarkResult> results) {
    final buffer = StringBuffer();
    buffer.writeln('# Inference Engine Benchmark Report');
    buffer.writeln('Generated: ${DateTime.now()}');
    buffer.writeln();

    results.sort((a, b) => a.avgInferenceTime.compareTo(b.avgInferenceTime));

    buffer.writeln('## Results (sorted by average inference time)');
    buffer.writeln();

    for (int i = 0; i < results.length; i++) {
      final result = results[i];
      buffer.writeln('### ${i + 1}. ${result.framework}');
      buffer.writeln('- **Average Inference Time**: ${result.avgInferenceTime.toStringAsFixed(2)} ms');
      buffer.writeln('- **Min Inference Time**: ${result.minInferenceTime.toStringAsFixed(2)} ms');
      buffer.writeln('- **Max Inference Time**: ${result.maxInferenceTime.toStringAsFixed(2)} ms');
      buffer.writeln('- **Memory Usage**: ${result.memoryUsage.toStringAsFixed(2)} MB');
      buffer.writeln('- **Iterations**: ${result.iterations}');
      buffer.writeln();
    }

    if (results.isNotEmpty) {
      final fastest = results.first;
      buffer.writeln('## Winner: ${fastest.framework}');
      buffer.writeln('Fastest average inference time: ${fastest.avgInferenceTime.toStringAsFixed(2)} ms');
    }

    return buffer.toString();
  }
}
