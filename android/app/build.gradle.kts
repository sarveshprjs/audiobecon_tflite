plugins {
    id("com.android.application")
    id("kotlin-android")
    id("dev.flutter.flutter-gradle-plugin")
}

android {
    namespace = "com.example.new_sound_app"
    compileSdk = flutter.compileSdkVersion
    ndkVersion = "27.0.12077973"

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_11
        targetCompatibility = JavaVersion.VERSION_11
    }

    kotlinOptions {
        jvmTarget = JavaVersion.VERSION_11.toString()
    }

    defaultConfig {
        applicationId = "com.example.new_sound_app"
        minSdk = 24
        targetSdk = flutter.targetSdkVersion
        versionCode = flutter.versionCode
        versionName = flutter.versionName

        ndk {
            abiFilters.addAll(listOf("armeabi-v7a", "arm64-v8a", "x86_64", "x86"))
        }
    }

    // ✅ UPDATED BUILD TYPES - R8 DISABLED TEMPORARILY
    buildTypes {
        release {
            signingConfig = signingConfigs.getByName("debug")
            isMinifyEnabled = false  // Changed from true to false
            isShrinkResources = false // Changed from true to false
            proguardFiles(
                getDefaultProguardFile("proguard-android.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            isMinifyEnabled = false
            isShrinkResources = false
        }
    }

    packagingOptions {
        pickFirst("lib/x86/libc++_shared.so")
        pickFirst("lib/x86_64/libc++_shared.so")
        pickFirst("lib/armeabi-v7a/libc++_shared.so")
        pickFirst("lib/arm64-v8a/libc++_shared.so")
    }
}

// ✅ ADD THESE DEPENDENCIES
dependencies {
    implementation("com.google.android.play:core:1.10.3")
    implementation("com.google.android.play:core-ktx:1.8.1")
    implementation("org.tensorflow:tensorflow-lite:2.4.0")
    implementation("org.tensorflow:tensorflow-lite-gpu:2.4.0")
}

flutter {
    source = "../.."
}

configurations.all {
    resolutionStrategy {
        force("org.jetbrains.kotlin:kotlin-stdlib:1.9.0")
        force("org.jetbrains.kotlin:kotlin-stdlib-jdk8:1.9.0")
    }
}