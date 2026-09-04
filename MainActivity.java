package com.sagnik.sensorrecorder;

import android.Manifest;
import android.content.pm.PackageManager;
import android.os.Build;
import android.os.Bundle;
import android.widget.Button;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.camera.view.PreviewView;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.util.ArrayList;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private static final int PERMISSION_REQUEST_CODE = 100;

    private Button btnRecord;
    private TextView tvStatus;
    private PreviewView previewView;
    private RecorderManager recorderManager;
    private boolean isRecording = false;

    private static final String[] REQUIRED_PERMISSIONS;

    static {
        List<String> perms = new ArrayList<>();
        perms.add(Manifest.permission.CAMERA);
        perms.add(Manifest.permission.RECORD_AUDIO);
        if (Build.VERSION.SDK_INT <= Build.VERSION_CODES.P) {
            perms.add(Manifest.permission.WRITE_EXTERNAL_STORAGE);
            perms.add(Manifest.permission.READ_EXTERNAL_STORAGE);
        }
        REQUIRED_PERMISSIONS = perms.toArray(new String[0]);
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        btnRecord = findViewById(R.id.btnRecord);
        tvStatus = findViewById(R.id.tvStatus);
        previewView = findViewById(R.id.previewView);

        recorderManager = new RecorderManager(this);

        btnRecord.setOnClickListener(v -> {
            if (!isRecording) {
                if (allPermissionsGranted()) {
                    startRecording();
                } else {
                    ActivityCompat.requestPermissions(this, REQUIRED_PERMISSIONS, PERMISSION_REQUEST_CODE);
                }
            } else {
                stopRecording();
            }
        });
    }

    private void startRecording() {
        isRecording = true;
        btnRecord.setText("STOP RECORDING");
        btnRecord.setBackgroundColor(0xFFE53935);
        tvStatus.setText("Starting...");

        recorderManager.startRecording(previewView, new RecorderManager.RecordingCallback() {
            @Override
            public void onStatusUpdate(String status) {
                runOnUiThread(() -> tvStatus.setText(status));
            }

            @Override
            public void onError(String error) {
                runOnUiThread(() -> {
                    tvStatus.setText("Error: " + error);
                    isRecording = false;
                    btnRecord.setText("START RECORDING");
                    btnRecord.setBackgroundColor(0xFF43A047);
                    Toast.makeText(MainActivity.this, "Error: " + error, Toast.LENGTH_LONG).show();
                });
            }
        });
    }

    private void stopRecording() {
        isRecording = false;
        btnRecord.setText("START RECORDING");
        btnRecord.setBackgroundColor(0xFF43A047);
        tvStatus.setText("Extracting frames from video, please wait...");

        recorderManager.stopRecording(outputPath -> runOnUiThread(() -> {
            tvStatus.setText("Saved!\n" + outputPath);
            Toast.makeText(this, "Recording saved!", Toast.LENGTH_LONG).show();
        }));
    }

    private boolean allPermissionsGranted() {
        for (String perm : REQUIRED_PERMISSIONS) {
            if (ContextCompat.checkSelfPermission(this, perm) != PackageManager.PERMISSION_GRANTED) {
                return false;
            }
        }
        return true;
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions,
                                           @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == PERMISSION_REQUEST_CODE && allPermissionsGranted()) {
            startRecording();
        } else {
            Toast.makeText(this, "Permissions required!", Toast.LENGTH_LONG).show();
        }
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (isRecording) recorderManager.stopRecording(null);
        recorderManager.release();
    }
}