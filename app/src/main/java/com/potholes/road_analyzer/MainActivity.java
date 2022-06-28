package com.potholes.road_analyzer;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;

import android.content.pm.PackageManager;
import android.content.res.AssetManager;

import android.os.Bundle;
import android.os.Environment;
import android.util.Log;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.Toast;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.JavaCameraView;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.*;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.MatOfRect2d;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Rect2d;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Net;
import org.opencv.imgproc.Imgproc;

import org.opencv.dnn.Dnn;
import org.opencv.utils.Converters;
import org.opencv.ximgproc.Ximgproc;


import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

public class MainActivity extends AppCompatActivity implements CameraBridgeViewBase.CvCameraViewListener2 {



    private static final int MY_PERMISSION_REQUEST_STORAGE = 1;
    CameraBridgeViewBase cameraBridgeViewBase;
    BaseLoaderCallback baseLoaderCallback;
    boolean startYolo = false;
    boolean firstTimeYolo = false;
    Net tinyYolo;
    Button detectButton;



    public void YOLO(View Button) throws IOException {


        if (startYolo == false){
            startYolo = true;
            detectButton.setText("Detection in progress");

            if (firstTimeYolo == false){

                firstTimeYolo = true;

                File f = new File(getCacheDir()+"/yolov3-tiny_obj.cfg");
                if(!f.exists()) try {

                    InputStream is = getAssets().open("yolov3-tiny_obj.cfg");
                    int size = is.available();
                    byte[] buffer = new byte[size];
                    is.read(buffer);
                    is.close();

                    FileOutputStream fos = new FileOutputStream(f);
                    fos.write(buffer);
                    fos.close();
                    Toast.makeText(this,"Config Saved", Toast.LENGTH_LONG).show();

                }catch (Exception e){
                    throw new RuntimeException(e);
                }

                File fw = new File(getCacheDir()+"/yolov3-tiny_obj_best.weights");
                if(!fw.exists()) try {

                    InputStream isw = getAssets().open("yolov3-tiny_obj_best.weights");
                    int sizew = isw.available();
                    byte[] bufferw = new byte[sizew];
                    isw.read(bufferw);
                    isw.close();

                    FileOutputStream fosw = new FileOutputStream(fw);
                    fosw.write(bufferw);
                    fosw.close();

                    Toast.makeText(this,"Model Saved", Toast.LENGTH_LONG).show();
                }catch (Exception e){
                    throw new RuntimeException(e);
                }

//                String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny_obj.cfg" ;
//                String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny_obj_best.weights";

                String tinyYoloCfg = f.getPath();
                String tinyYoloWeights = fw.getPath();

                tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);


            }
 }

        else{

            startYolo = false;
            detectButton.setText("Detect");


        }

    }






    @Override
    protected void onCreate(Bundle savedInstanceState) {

        super.onCreate(savedInstanceState);
        //copyAssets("yolov3-tiny_obj.cfg");
        //copyAssets("yolov3-tiny_obj_best.weights");
        setContentView(R.layout.activity_main);

        cameraBridgeViewBase = (JavaCameraView)findViewById(R.id.CameraView);
        cameraBridgeViewBase.setVisibility(SurfaceView.VISIBLE);
        cameraBridgeViewBase.setCvCameraViewListener(this);

        detectButton = (Button) findViewById(R.id.button3);




        // Give app permissions
        if(ContextCompat.checkSelfPermission(MainActivity.this,
                Manifest.permission.WRITE_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {

            if(ActivityCompat.shouldShowRequestPermissionRationale(MainActivity.this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE)) {
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE}, MY_PERMISSION_REQUEST_STORAGE);
            }else {
                ActivityCompat.requestPermissions(MainActivity.this,
                        new String[] {Manifest.permission.WRITE_EXTERNAL_STORAGE}, MY_PERMISSION_REQUEST_STORAGE);
            }

        }else {
            // Do nothing
        }

        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        baseLoaderCallback = new BaseLoaderCallback(this) {
            @Override
            public void onManagerConnected(int status) {
                super.onManagerConnected(status);

                switch(status){

                    case BaseLoaderCallback.SUCCESS:
                        cameraBridgeViewBase.enableView();
                        break;
                    default:
                        super.onManagerConnected(status);
                        break;
                }

            }

        };


    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat frame = inputFrame.rgba();

            if (startYolo == true) {

            Imgproc.cvtColor(frame, frame, Imgproc.COLOR_RGBA2RGB);

            Mat imageBlob = Dnn.blobFromImage(frame, 0.00392, new Size(416,416),new Scalar(0, 0, 0),/*swapRB*/false, /*crop*/false);
            tinyYolo.setInput(imageBlob);

            java.util.List<Mat> result = new java.util.ArrayList<Mat>(2);

            List<String> outBlobNames = new java.util.ArrayList<>();
            outBlobNames.add(0, "yolo_16");
            outBlobNames.add(1, "yolo_23");

            tinyYolo.forward(result,outBlobNames);
            float confThreshold = 0.1f;


            List<Integer> clsIds = new ArrayList<>();
            List<Float> confs = new ArrayList<>();
            List<Rect2d> rects = new ArrayList<>();

            for (int i = 0; i < result.size(); ++i)
            {

                Mat level = result.get(i);

                for (int j = 0; j < level.rows(); ++j)
                {
                    Mat row = level.row(j);
                    Mat scores = row.colRange(5, level.cols());

                    Core.MinMaxLocResult mm = Core.minMaxLoc(scores);
                    float confidence = (float)mm.maxVal;
                    Point classIdPoint = mm.maxLoc;

                    if (confidence > confThreshold)
                    {
                        int centerX = (int)(row.get(0,0)[0] * frame.cols());
                        int centerY = (int)(row.get(0,1)[0] * frame.rows());
                        int width   = (int)(row.get(0,2)[0] * frame.cols());
                        int height  = (int)(row.get(0,3)[0] * frame.rows());


                        int left    = centerX - width  / 2;
                        int top     = centerY - height / 2;

                        clsIds.add((int)classIdPoint.x);
                        confs.add((float)confidence);




                        rects.add(new Rect2d(left, top, width, height));
                    }
                }
            }
            int ArrayLength = confs.size();

            if (ArrayLength>=1) {
                // Apply non-maximum suppression procedure.
                float nmsThresh = 0.2f;

                MatOfFloat confidences = new MatOfFloat(Converters.vector_float_to_Mat(confs));
                Rect2d[] boxesArray = rects.toArray(new Rect2d[0]);

                MatOfRect2d boxes = new MatOfRect2d(boxesArray);

                MatOfInt indices = new MatOfInt();
                Dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThresh, indices);


                // Draw result boxes:
                int[] ind = indices.toArray();
                for (int i = 0; i < ind.length; ++i) {

                    int idx = ind[i];
                    Rect2d box = boxesArray[idx];

                    int idGuy = clsIds.get(idx);

                    float conf = confs.get(idx);


                    List<String> cocoNames = Arrays.asList("Affaissement | Trace des roues", "Affaissement | Jointure de construction", "Affaissement | Interval égal", "Affaissement | Jointure de construction", "Affaissement | Alligator crack","Affaissement | Nid de poule","Passage pour piétons","Ligne blanche");
                    int intConf = (int) (conf * 100);
                    Imgproc.putText(frame,cocoNames.get(idGuy) + " " + intConf + "%",box.tl(),Core.FONT_ITALIC, 1, new Scalar(255,255,0),2);
                    Imgproc.rectangle(frame, box.tl(), box.br(), new Scalar(255, 0, 0), 1);


                }
            }


        }



        return frame;
    }


    @Override
    public void onCameraViewStarted(int width, int height) {


        if (startYolo == true){


            File f = new File(getCacheDir()+"/yolov3-tiny_obj.cfg");
            if(!f.exists()) try {

                InputStream is = getAssets().open("yolov3-tiny_obj.cfg");
                int size = is.available();
                byte[] buffer = new byte[size];
                is.read(buffer);
                is.close();

                FileOutputStream fos = new FileOutputStream(f);
                fos.write(buffer);
                fos.close();

            }catch (Exception e){
                throw new RuntimeException(e);
            }

            File fw = new File(getCacheDir()+"/yolov3-tiny_obj_best.weights");
            if(!fw.exists()) try {

                InputStream isw = getAssets().open("yolov3-tiny_obj_best.weights");
                int sizew = isw.available();
                byte[] bufferw = new byte[sizew];
                isw.read(bufferw);
                isw.close();

                FileOutputStream fosw = new FileOutputStream(fw);
                fosw.write(bufferw);
                fosw.close();

            }catch (Exception e){
                throw new RuntimeException(e);
            }

//                String tinyYoloCfg = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny_obj.cfg" ;
//                String tinyYoloWeights = Environment.getExternalStorageDirectory() + "/dnns/yolov3-tiny_obj_best.weights";

            String tinyYoloCfg = f.getPath();
            String tinyYoloWeights = fw.getPath();

            tinyYolo = Dnn.readNetFromDarknet(tinyYoloCfg, tinyYoloWeights);
//philippe voyer christophe baudot

        }



    }


    @Override
    public void onCameraViewStopped() {

    }


    @Override
    protected void onResume() {
        super.onResume();

        if (!OpenCVLoader.initDebug()){
            Toast.makeText(getApplicationContext(),"There's a problem, yo!", Toast.LENGTH_SHORT).show();
        }

        else
        {
            baseLoaderCallback.onManagerConnected(baseLoaderCallback.SUCCESS);
        }



    }

    @Override
    protected void onPause() {
        super.onPause();
        if(cameraBridgeViewBase!=null){

            cameraBridgeViewBase.disableView();
        }

    }


    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (cameraBridgeViewBase!=null){
            cameraBridgeViewBase.disableView();
        }
    }




                @Override
                public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
                    super.onRequestPermissionsResult(requestCode, permissions, grantResults);
                    switch (requestCode) {
                        case MY_PERMISSION_REQUEST_STORAGE: {
                            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                                if (ContextCompat.checkSelfPermission(MainActivity.this,
                                        Manifest.permission.WRITE_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                                    }
                                }
                                else
                                    {
                                    Toast.makeText(this, "No permission Granted", Toast.LENGTH_SHORT).show();
                                }

                        }
                    }
                }


        private void copyAssets(String filename) {

        String dirPath = Environment.getExternalStorageDirectory().getAbsolutePath() + "/dnns";
        File dir = new File(dirPath);
        if(!dir.exists()){
            dir.mkdir();
        }

        AssetManager assetManager = getAssets();
        InputStream in = null;
        OutputStream out = null;

            try {
                in = assetManager.open(filename);
                File outFile = new File(dirPath, filename);
                out = new FileOutputStream(outFile);
                copyFile(in, out);
                in.close();
                Toast.makeText(this,"Model Saved", Toast.LENGTH_LONG).show();

            } catch(IOException e) {
                Log.e("tag", "Failed to copy asset file: " + filename, e);
                Toast.makeText(this,"Failed", Toast.LENGTH_SHORT).show();

            }finally {
                if(in != null) {
                    try{
                        in.close();
                    } catch(IOException e){
                        e.printStackTrace();
                    }

                    }

                if(out != null) {
                    try{
                        out.close();
                    } catch(IOException e){
                        e.printStackTrace();
                    }

                }


                }
            }




    private void copyFile(InputStream in, OutputStream out) throws IOException {
        byte[] buffer = new byte[1024];
        int read;
        while((read = in.read(buffer)) != -1){
            out.write(buffer, 0, read);
        }
    }


}