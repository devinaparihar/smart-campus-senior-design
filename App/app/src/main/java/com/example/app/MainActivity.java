package com.example.app;

import androidx.appcompat.app.AppCompatActivity;

import android.app.Application;
import android.os.Bundle;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;

public class MainActivity extends AppCompatActivity implements OnClickListener{

    private Button msgButton;
    private AmplifyApp amplifyApp;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        amplifyApp = AmplifyApp.getInstance();
        msgButton = (Button)findViewById(R.id.msgButton);
        msgButton.setOnClickListener(this);
    }

    @Override
    public void onClick(View v) {
        amplifyApp.sendMessage("This is a test");
    }
}