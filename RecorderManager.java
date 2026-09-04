package com.sagnik.sensorrecorder;

import android.content.Context;
import android.graphics.Bitmap;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.media.MediaMetadataRetriever;
import android.os.Environment;
import android.util.Log;
import android.util.Size;

import androidx.camera.core.Camera;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.Preview;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.camera.video.FileOutputOptions;
import androidx.camera.video.Quality;
import androidx.camera.video.QualitySelector;
import androidx.camera.video.Recorder;
import androidx.camera.video.Recording;
import androidx.camera.video.VideoCapture;
import androidx.camera.video.VideoRecordEvent;
import androidx.camera.view.PreviewView;
import androidx.core.content.ContextCompat;
import androidx.lifecycle.LifecycleOwner;

import com.google.common.util.concurrent.ListenableFuture;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class RecorderManager {

    private static final String TAG = "RecorderManager";
    private static final long FRAME_INTERVAL_US = 33333; // ~30fps in microseconds

    private final Context context;
    private ExecutorService executor;

    // CameraX
    private ProcessCameraProvider cameraProvider;
    private VideoCapture<Recorder> videoCapture;
    private Recording activeRecording;
    private File videoFile;

    // Sensors
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private Sensor gyroscope;
    private SensorEventListener accelListener;
    private SensorEventListener gyroListener;

    // Output
    private File outputDir;
    private File framesDir;
    private BufferedWriter accelWriter;
    private BufferedWriter gyroWriter;
    private volatile boolean isRecording = false;
    private long baseNano;
    private long baseMillis;

    public interface RecordingCallback {
        void onStatusUpdate(String status);
        void onError(String error);
    }

    public interface StopCallback {
        void onStopped(String outputPath);
    }

    public RecorderManager(Context context) {
        this.context = context;
        sensorManager = (SensorManager) context.getSystemService(Context.SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);
        gyroscope = sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE);
        executor = Executors.newSingleThreadExecutor();
    }

    public void startRecording(PreviewView previewView, RecordingCallback callback) {
        try {
            outputDir = createOutputDirectory();
            framesDir = new File(outputDir, "frames");
            framesDir.mkdirs();

            accelWriter = new BufferedWriter(new FileWriter(new File(outputDir, "accelerometer.csv")));
            gyroWriter  = new BufferedWriter(new FileWriter(new File(outputDir, "gyroscope.csv")));
            accelWriter.write("timestamp_ns,x,y,z\n");
            gyroWriter.write("timestamp_ns,x,y,z\n");

        } catch (IOException e) {
            callback.onError("Failed to create output files: " + e.getMessage());
            return;
        }

        isRecording = true;

// ADD THESE 2 LINES
        baseNano = System.nanoTime();
        baseMillis = System.currentTimeMillis();

        startSensors();

        ListenableFuture<ProcessCameraProvider> future =
                ProcessCameraProvider.getInstance(context);

        future.addListener(() -> {
            try {
                cameraProvider = future.get();
                bindCamera(previewView, callback);
            } catch (Exception e) {
                Log.e(TAG, "CameraProvider failed", e);
                callback.onError("Camera init failed: " + e.getMessage());
            }
        }, ContextCompat.getMainExecutor(context));
    }

    private void bindCamera(PreviewView previewView, RecordingCallback callback) {
        try {
            cameraProvider.unbindAll();

            Preview preview = new Preview.Builder()
                    .setTargetResolution(new Size(640, 480))
                    .build();
            preview.setSurfaceProvider(previewView.getSurfaceProvider());

            Recorder recorder = new Recorder.Builder()
                    .setQualitySelector(QualitySelector.from(Quality.SD))
                    .build();
            videoCapture = VideoCapture.withOutput(recorder);

            // Only Preview + VideoCapture — no ImageAnalysis competing for camera
            Camera camera = cameraProvider.bindToLifecycle(
                    (LifecycleOwner) context,
                    CameraSelector.DEFAULT_BACK_CAMERA,
                    preview,
                    videoCapture
            );

            videoFile = new File(outputDir, "video.mp4");
            FileOutputOptions options = new FileOutputOptions.Builder(videoFile).build();

            activeRecording = videoCapture.getOutput()
                    .prepareRecording(context, options)
                    .withAudioEnabled()
                    .start(ContextCompat.getMainExecutor(context), event -> {
                        if (event instanceof VideoRecordEvent.Start) {
                            callback.onStatusUpdate("Recording...");
                        } else if (event instanceof VideoRecordEvent.Finalize) {
                            VideoRecordEvent.Finalize fin = (VideoRecordEvent.Finalize) event;
                            if (fin.hasError()) {
                                Log.e(TAG, "Video finalize error: " + fin.getCause());
                            }
                        }
                    });

        } catch (Exception e) {
            Log.e(TAG, "bindCamera failed", e);
            callback.onError("Camera bind failed: " + e.getMessage());
        }
    }

    public void stopRecording(StopCallback callback) {
        isRecording = false;
        stopSensors();
        closeWriters();

        try {
            if (activeRecording != null) {
                activeRecording.stop();
                activeRecording = null;
            }
        } catch (Exception e) {
            Log.e(TAG, "Error stopping recording", e);
        }

        try {
            if (cameraProvider != null) {
                cameraProvider.unbindAll();
            }
        } catch (Exception e) {
            Log.e(TAG, "Error unbinding camera", e);
        }

        // Extract frames from video on background thread
        // Show "Extracting frames..." while this runs
        executor.execute(() -> {
            extractFramesFromVideo();
            if (callback != null && outputDir != null) {
                callback.onStopped(outputDir.getAbsolutePath());
            }
        });
    }

    // ─── FRAME EXTRACTION FROM VIDEO ──────────────────────────────────────────

    private void extractFramesFromVideo() {
        // Wait a moment for the video file to be fully written
        try { Thread.sleep(1000); } catch (InterruptedException ignored) {}

        if (videoFile == null || !videoFile.exists()) {
            Log.e(TAG, "Video file not found for frame extraction");
            return;
        }

        MediaMetadataRetriever retriever = new MediaMetadataRetriever();
        try {
            retriever.setDataSource(videoFile.getAbsolutePath());

            String durationStr = retriever.extractMetadata(
                    MediaMetadataRetriever.METADATA_KEY_DURATION);
            if (durationStr == null) {
                Log.e(TAG, "Could not get video duration");
                return;
            }

            long durationMs = Long.parseLong(durationStr);
            long durationUs = durationMs * 1000;

            Log.d(TAG, "Extracting frames from " + durationMs + "ms video");

            int frameCount = 0;
            for (long timeUs = 0; timeUs < durationUs; timeUs += FRAME_INTERVAL_US) {
                Bitmap frame = retriever.getFrameAtTime(
                        timeUs,
                        MediaMetadataRetriever.OPTION_CLOSEST
                );
                if (frame == null) continue;

                long timestampMs = baseMillis + (timeUs / 1000);
                File frameFile = new File(framesDir, timestampMs + ".jpg");

                try (FileOutputStream fos = new FileOutputStream(frameFile)) {
                    frame.compress(Bitmap.CompressFormat.JPEG, 85, fos);
                } catch (IOException e) {
                    Log.e(TAG, "Frame write error at " + timestampMs, e);
                } finally {
                    frame.recycle();
                }
                frameCount++;
            }

            Log.d(TAG, "Done! Extracted " + frameCount + " frames");

        } catch (Exception e) {
            Log.e(TAG, "Frame extraction failed", e);
        } finally {
            try { retriever.release(); } catch (Exception ignored) {}
        }
    }

    // ─── SENSORS ──────────────────────────────────────────────────────────────

    private void startSensors() {
        int delayUs = 5000; // request 200Hz

        accelListener = new SensorEventListener() {
            @Override
            public void onSensorChanged(SensorEvent event) {
                if (!isRecording) return;
                try {
                    long timestampMs = baseMillis + (event.timestamp - baseNano) / 1_000_000;

                    accelWriter.write(timestampMs + "," +
                            event.values[0] + "," +
                            event.values[1] + "," +
                            event.values[2] + "\n");
                } catch (IOException e) {
                    Log.e(TAG, "Accel write error", e);
                }
            }
            @Override public void onAccuracyChanged(Sensor sensor, int accuracy) {}
        };

        gyroListener = new SensorEventListener() {
            @Override
            public void onSensorChanged(SensorEvent event) {
                if (!isRecording) return;
                try {
                    long timestampMs = baseMillis + (event.timestamp - baseNano) / 1_000_000;

                    gyroWriter.write(timestampMs + "," +
                            event.values[0] + "," +
                            event.values[1] + "," +
                            event.values[2] + "\n");
                } catch (IOException e) {
                    Log.e(TAG, "Gyro write error", e);
                }
            }
            @Override public void onAccuracyChanged(Sensor sensor, int accuracy) {}
        };

        if (accelerometer != null)
            sensorManager.registerListener(accelListener, accelerometer, delayUs);
        if (gyroscope != null)
            sensorManager.registerListener(gyroListener, gyroscope, delayUs);
    }

    private void stopSensors() {
        if (accelListener != null) sensorManager.unregisterListener(accelListener);
        if (gyroListener != null) sensorManager.unregisterListener(gyroListener);
    }

    // ─── HELPERS ──────────────────────────────────────────────────────────────

    private File createOutputDirectory() {
        String timestamp = new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date());
        File base = new File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                "SensorRecorder"
        );
        File dir = new File(base, "Recording_" + timestamp);
        dir.mkdirs();
        return dir;
    }

    private void closeWriters() {
        try { if (accelWriter != null) { accelWriter.flush(); accelWriter.close(); accelWriter = null; } }
        catch (IOException e) { Log.e(TAG, "accelWriter close error", e); }
        try { if (gyroWriter != null) { gyroWriter.flush(); gyroWriter.close(); gyroWriter = null; } }
        catch (IOException e) { Log.e(TAG, "gyroWriter close error", e); }
    }

    public void release() {
        if (isRecording) stopRecording(null);
        executor.shutdown();
    }
}