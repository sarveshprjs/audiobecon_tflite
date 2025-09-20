# Flutter
-keep class io.flutter.app.** { *; }
-keep class io.flutter.plugin.** { *; }
-keep class io.flutter.util.** { *; }
-keep class io.flutter.view.** { *; }
-keep class io.flutter.** { *; }
-keep class io.flutter.plugins.** { *; }

# TensorFlow Lite
-keep class org.tensorflow.lite.** { *; }
-keep class org.tensorflow.** { *; }
-keep class org.tensorflow.lite.gpu.** { *; }
-keep class org.tensorflow.lite.support.** { *; }
-keep class org.tensorflow.lite.task.** { *; }

# Play Core
-keep class com.google.android.play.core.** { *; }
-keep class com.google.android.play.** { *; }
-keep class com.google.android.play.core.splitcompat.** { *; }
-keep class com.google.android.play.core.splitinstall.** { *; }
-keep class com.google.android.play.core.tasks.** { *; }

# Required for reflection
-keep class androidx.lifecycle.ViewModelProvider$Factory { *; }
-keep class androidx.lifecycle.ViewModel { *; }

# Keep native methods
-keepclasseswithmembernames class * {
    native <methods>;
}

# Keep annotations
-keepattributes *Annotation*

# Keep generic types
-keepattributes Signature

# Keep GpuDelegate classes specifically
-keep class org.tensorflow.lite.gpu.GpuDelegate { *; }
-keep class org.tensorflow.lite.gpu.GpuDelegateFactory { *; }
-keep class org.tensorflow.lite.gpu.GpuDelegate$Options { *; }