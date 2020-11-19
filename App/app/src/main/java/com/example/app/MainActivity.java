package com.example.app;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.NotificationCompat;

import android.app.AlarmManager;
import android.app.Application;
import android.app.PendingIntent;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.os.SystemClock;
import android.text.util.Linkify;
import android.util.Log;
import android.view.View;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.CheckBox;
import android.widget.CompoundButton;
import android.widget.EditText;
import android.widget.TextView;
import android.widget.ToggleButton;

import java.util.Timer;
import java.util.TimerTask;

public class MainActivity extends AppCompatActivity implements OnClickListener{

    private Button msgButton;
    private AmplifyApp amplifyApp;


    //instance variables from find3
    // logging
    private final String TAG = "MainActivity";

    // background manager
    private PendingIntent recurringLl24 = null;
    private Intent ll24 = null;
    AlarmManager alarms = null;
    Timer timer = null;
    private RemindTask oneSecondTimer = null;

    private String[] autocompleteLocations = new String[] {"bedroom","living room","kitchen","bathroom", "office"};

    class RemindTask extends TimerTask {
        private Integer counter = 0;

        public void resetCounter() {
            counter = 0;
        }

        public void run() {
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    counter++;
                    TextView rssi_msg = (TextView) findViewById(R.id.textOutput);
                    String currentText = rssi_msg.getText().toString();
                    if (currentText.contains("ago: ")) {
                        String[] currentTexts = currentText.split("ago: ");
                        currentText = currentTexts[1];
                    }
                    rssi_msg.setText(counter + " seconds ago: " + currentText);
                }
            });
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        amplifyApp = AmplifyApp.getInstance();
        msgButton = (Button)findViewById(R.id.msgButton);
        msgButton.setOnClickListener(this);

        //Code Below derived from find3 by Schullz

        ToggleButton toggleButton = (ToggleButton) findViewById(R.id.toggleButton);
        toggleButton.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton buttonView, boolean isChecked) {
                if (isChecked) {
                    TextView rssi_msg = (TextView) findViewById(R.id.textOutput);
                    /*
                    String familyName = ((EditText) findViewById(R.id.familyName)).getText().toString().toLowerCase();
                    if (familyName.equals("")) {
                        rssi_msg.setText("family name cannot be empty");
                        buttonView.toggle();
                        return;
                    }

                    String deviceName = ((EditText) findViewById(R.id.deviceName)).getText().toString().toLowerCase();
                    if (deviceName.equals("")) {
                        rssi_msg.setText("device name cannot be empty");
                        buttonView.toggle();
                        return;
                    }
                    */
                    String familyName = "family";
                    String deviceName = "device";
                    String serverAddress = "address";
                    boolean allowGPS = ((CheckBox) findViewById(R.id.allowGPS)).isChecked();
                    Log.d(TAG,"allowGPS is checked: "+allowGPS);
                /*
                    String locationName = ((EditText) findViewById(R.id.locationName)).getText().toString().toLowerCase();

                    CompoundButton trackingButton = (CompoundButton) findViewById(R.id.toggleScanType);
                    if (trackingButton.isChecked() == false) {
                        locationName = "";
                    } else {
                        if (locationName.equals("")) {
                            rssi_msg.setText("location name cannot be empty when learning");
                            buttonView.toggle();
                            return;
                        }
                    }
                */
                    String locationName = "location";

                    SharedPreferences sharedPref = MainActivity.this.getPreferences(Context.MODE_PRIVATE);
                    SharedPreferences.Editor editor = sharedPref.edit();
                    editor.putString("familyName", familyName);
                    editor.putString("deviceName", deviceName);
                    editor.putString("serverAddress", serverAddress);
                    editor.putString("locationName", locationName);
                    editor.putBoolean("allowGPS",allowGPS);
                    editor.commit();

                    rssi_msg.setText("running");
                    // 24/7 alarm
                    ll24 = new Intent(MainActivity.this, AlarmReceiverLife.class);
                    Log.d(TAG, "setting familyName to [" + familyName + "]");
                    ll24.putExtra("familyName", familyName);
                    ll24.putExtra("deviceName", deviceName);
                    ll24.putExtra("serverAddress", serverAddress);
                    ll24.putExtra("locationName", locationName);
                    ll24.putExtra("allowGPS",allowGPS);
                    recurringLl24 = PendingIntent.getBroadcast(MainActivity.this, 0, ll24, PendingIntent.FLAG_CANCEL_CURRENT);
                    alarms = (AlarmManager) getSystemService(Context.ALARM_SERVICE);
                    alarms.setRepeating(AlarmManager.RTC_WAKEUP, SystemClock.currentThreadTimeMillis(), 60000, recurringLl24);
                    timer = new Timer();
                    oneSecondTimer = new RemindTask();
                    timer.scheduleAtFixedRate(oneSecondTimer, 1000, 1000);

                    String scanningMessage = "Scanning for " + familyName + "/" + deviceName;
                    if (locationName.equals("") == false) {
                        scanningMessage += " at " + locationName;
                    }
//                    NotificationCompat.Builder notificationBuilder = new NotificationCompat.Builder(MainActivity.this)
//                            .setSmallIcon(R.drawable.ic_stat_name)
//                            .setContentTitle(scanningMessage)
//                            .setContentIntent(recurringLl24);
                    //specifying an action and its category to be triggered once clicked on the notification
                    Intent resultIntent = new Intent(MainActivity.this, MainActivity.class);
                    resultIntent.setAction("android.intent.action.MAIN");
                    resultIntent.addCategory("android.intent.category.LAUNCHER");
                    PendingIntent resultPendingIntent = PendingIntent.getActivity(MainActivity.this, 0, resultIntent, PendingIntent.FLAG_UPDATE_CURRENT);
//                    notificationBuilder.setContentIntent(resultPendingIntent);
//
//                    android.app.NotificationManager notificationManager =
//                            (android.app.NotificationManager) MainActivity.this.getSystemService(Context.NOTIFICATION_SERVICE);
//                    notificationManager.notify(0 /* ID of notification */, notificationBuilder.build());


//                    final TextView myClickableUrl = (TextView) findViewById(R.id.textInstructions);
//                    myClickableUrl.setText("See your results in realtime: " + serverAddress + "/view/location/" + familyName + "/" + deviceName);
//                    Linkify.addLinks(myClickableUrl, Linkify.WEB_URLS);
                } else {
                    TextView rssi_msg = (TextView) findViewById(R.id.textOutput);
                    rssi_msg.setText("not running");
                    Log.d(TAG, "toggle set to false");
                    alarms.cancel(recurringLl24);
                    android.app.NotificationManager mNotificationManager = (android.app.NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
                    mNotificationManager.cancel(0);
                    timer.cancel();
                }
            }
        });
    }

    @Override
    protected void onDestroy() {
        Log.d(TAG, "MainActivity onDestroy()");
        if (alarms != null) alarms.cancel(recurringLl24);
        if (timer != null) timer.cancel();
        android.app.NotificationManager mNotificationManager = (android.app.NotificationManager) getSystemService(Context.NOTIFICATION_SERVICE);
        mNotificationManager.cancel(0);
        Intent scanService = new Intent(this, ScanService.class);
        stopService(scanService);
        super.onDestroy();
    }

    @Override
    public void onClick(View v) {
        amplifyApp.sendMessage("Test number 2");
    }
}