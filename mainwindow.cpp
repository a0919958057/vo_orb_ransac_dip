#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <opencv2/opencv.hpp>
#include <QTimer>
using namespace cv;

// Calculates rotation matrix given euler angles.
Mat eulerAnglesToRotationMatrix(Vec3f &theta)
{
    // Calculate rotation about x axis
    Mat R_x = (Mat_<double>(3,3) <<
               1,       0,              0,
               0,       cos(theta[0]),   -sin(theta[0]),
               0,       sin(theta[0]),   cos(theta[0])
               );

    // Calculate rotation about y axis
    Mat R_y = (Mat_<double>(3,3) <<
               cos(theta[1]),    0,      sin(theta[1]),
               0,               1,      0,
               -sin(theta[1]),   0,      cos(theta[1])
               );

    // Calculate rotation about z axis
    Mat R_z = (Mat_<double>(3,3) <<
               cos(theta[2]),    -sin(theta[2]),      0,
               sin(theta[2]),    cos(theta[2]),       0,
               0,               0,                  1);


    // Combined rotation matrix
    Mat R = R_z * R_y * R_x;
    std::cout << "R = " << R << std::endl;
    return R;

}

Vec3f rotationMatrixToEulerAngles(Mat &R)
{

    float sy = sqrt(R.at<double>(0,0) * R.at<double>(0,0) +  R.at<double>(1,0) * R.at<double>(1,0) );

    bool singular = sy < 1e-6; // If

    float x, y, z;
    if (!singular)
    {
        x = atan2(R.at<double>(2,1) , R.at<double>(2,2));
        y = atan2(-R.at<double>(2,0), sy);
        z = atan2(R.at<double>(1,0), R.at<double>(0,0));
    }
    else
    {
        x = atan2(-R.at<double>(1,2), R.at<double>(1,1));
        y = atan2(-R.at<double>(2,0), sy);
        z = 0;
    }
    return Vec3f(x, y, z);



}


void MainWindow::calib_steo()
{
    std::vector<std::vector<Point2f>> r_image_points;
    std::vector<std::vector<Point2f>> l_image_points;
    std::vector<std::vector<Point3f>> object_points;
    char buffer1[100];
    char buffer2[100];
    for (int count = 1; count < 26; count++) {


        sprintf(buffer1, "./sample2/left/image_l_%d.jpg", count);
        Mat image_samp1 = imread(buffer1, CV_LOAD_IMAGE_GRAYSCALE);
        printf("Read %s image\n", buffer1);

        sprintf(buffer2, "./sample2/right/image_r_%d.jpg", count);
        Mat image_samp2 = imread(buffer2, CV_LOAD_IMAGE_GRAYSCALE);
        printf("Read %s image\n", buffer2);



        // Find left image corner
        Mat l_corners;
        bool patternfound_left = findChessboardCorners(image_samp1, Size(7, 5), l_corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

        if (patternfound_left)
            cornerSubPix(image_samp1, l_corners, Size(11, 11), Size(-1, -1),
                TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));


        // Find right image corner
        Mat r_corners;
        bool patternfound_right = findChessboardCorners(image_samp2, Size(7, 5), r_corners, CALIB_CB_ADAPTIVE_THRESH + CALIB_CB_NORMALIZE_IMAGE
            + CALIB_CB_FAST_CHECK);

        if (patternfound_right)
            cornerSubPix(image_samp2, r_corners, Size(11, 11), Size(-1, -1),
                TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));


        Size boarder_sz(7, 5);
        std::vector<Point3f> dstCandidateCorners;
        dstCandidateCorners.clear();
        for (int i = 0; i < boarder_sz.height; i++) {
            for (int j = 0; j < boarder_sz.width; j++) {
                dstCandidateCorners.push_back(
                    Point3f(static_cast<float>(i)*33.5, static_cast<float>(j)*33.5, 0.0f));
            }
        }

        l_image_points.push_back(l_corners);
        r_image_points.push_back(r_corners);

        object_points.push_back(dstCandidateCorners);

    }


    double rms = stereoCalibrate(object_points, l_image_points, r_image_points,
            coff_k1, coff_d1,
            coff_k2, coff_d2,
            Size(640, 480), r, t, e, f,
            CV_CALIB_USE_INTRINSIC_GUESS,
            TermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 30, 1e-5)
            );

    Rect validRoi[2];
        stereoRectify(coff_k1, coff_d1, coff_k2, coff_d2, Size(640, 480),
            r, t, R1, R2, P1, P2, Q, CV_CALIB_ZERO_DISPARITY, 0, Size(640, 480), &validRoi[0], &validRoi[1]);

    initUndistortRectifyMap(coff_k1, coff_d1, R1, P1, Size(640, 480), CV_32FC1,
        map_l_1,
        map_l_2);

    initUndistortRectifyMap(coff_k2, coff_d2, R2, P2, Size(640, 480), CV_32FC1,
        map_r_1,
        map_r_2);
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow),
    cap_left(2),
    cap_right(1),
    KF(6, 6, 0)
{
    ui->setupUi(this);

    canvas_raw.create(240, 320 * 2, CV_8UC3);
    canvas.create(240, 320 * 2, CV_8UC3);
    timer = new QTimer(this);
    timer->start(33);


    FileStorage rgb_l_setting("./camera_l.yml", CV_STORAGE_READ);
    rgb_l_setting["camera_matrix"] >> coff_k1;
    rgb_l_setting["distortion_coefficients"] >> coff_d1;
    rgb_l_setting.release();

    FileStorage rgb_r_setting("./camera_r.yml", CV_STORAGE_READ);
    rgb_r_setting["camera_matrix"] >> coff_k2;
    rgb_r_setting["distortion_coefficients"] >> coff_d2;
    rgb_r_setting.release();

    calib_steo();

    u0_l = P1.at<double>(0, 2);
    u0_r = P2.at<double>(0, 2);
    //float u0 = 340;
    v0_l = P1.at<double>(1, 2);
    v0_r = P2.at<double>(1, 2);

    fu_l = P1.at<double>(0, 0);
    fu_r = P2.at<double>(0, 0);

    fv_l = P1.at<double>(1, 1);
    fv_r = P2.at<double>(1, 1);

    // Setup KalmanFilter
    setIdentity(KF.transitionMatrix);
    setIdentity(KF.measurementMatrix);
    setIdentity(KF.processNoiseCov, Scalar::all(10.0));
    setIdentity(KF.measurementNoiseCov, Scalar::all(25.0));
    setIdentity(KF.errorCovPost, Scalar::all(1));
    measure = Mat::zeros(6, 1, CV_32F);

    // Initialize Pose state matrix
    pose_state = Mat::eye(4,4, CV_64FC1);
    //Mat roi_r = pose_state(Rect(0,0,3,3));
    //roi_r = Mat::eye(3,3,CV_64FC1);
    std::cout << "Homogeneous matrix = " << pose_state << std::endl;
    L = -P2.at<double>(0, 3) / P2.at<double>(0, 0);

    is_kpinit = false;

    id_counter = 0;
    detector = ORB::create(15, 2, 4, 31, 0, 2 , ORB::HARRIS_SCORE, 31, 10);
    //detector = ORB::create(100, 2.0, 8, 80);
    ui->featureCountNumber->display(100);

    connect(timer, SIGNAL(timeout()),this,SLOT(showImage()));
}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow::showImage() {

    cap_left >> image1;
    cap_right >> image2;

    Mat canvas_left = canvas_raw(Rect(0, 0, 320, 240));
    Mat canvas_right = canvas_raw(Rect(320, 0, 320, 240));

    Mat canvasPart1 = canvas(Rect(0, 0, 320, 240));
    Mat canvasPart2 = canvas(Rect(320, 0, 320, 240));

    cv::resize(image1, canvas_left, canvas_left.size(), 0, 0, INTER_AREA);
    cv::resize(image2, canvas_right, canvas_right.size(), 0, 0, INTER_AREA);

    remap(image1, image1, map_l_1, map_l_2, CV_INTER_LINEAR);
    remap(image2, image2, map_r_1, map_r_2, CV_INTER_LINEAR);



    std::vector<KeyPoint> keyPoints_l, keyPoints_r;
    Mat dcpt_l, dcpt_r;

//    detector->detectAndCompute(image1, noArray(), keyPoints_l, dcpt_l);
//    detector->detectAndCompute(image2, noArray(), keyPoints_r, dcpt_r);

#define ROI_V_DIV 1
#define ROI_U_DIV4 1
    for (int i = 0; i < ROI_U_DIV4; i++) {
      for (int j = 0; j < ROI_V_DIV; j++) {
        std::vector<KeyPoint> keyPoints_test;
        Mat dcpt_test;

        Mat roi_image = image1(Rect(i * image1.cols / ROI_U_DIV4, j * image1.rows / ROI_V_DIV, image1.cols / ROI_U_DIV4, image1.rows / ROI_V_DIV));

        detector->detectAndCompute(roi_image, noArray(), keyPoints_test, dcpt_test);
//        int kp_size_error(ref_kp_size - keyPoints_test.size());
//        detector->getFastThreshold();


        for (int k = 0; k < keyPoints_test.size(); k++) {
          keyPoints_test[k].pt = Point(i * image1.cols / ROI_U_DIV4 + keyPoints_test[k].pt.x, j * image1.rows / ROI_V_DIV + keyPoints_test[k].pt.y);
          keyPoints_l.push_back(keyPoints_test[k]);
          dcpt_l.push_back(dcpt_test.row(k));
        }
      }
    }
    for (int i = 0; i < ROI_U_DIV4; i++) {
      for (int j = 0; j < ROI_V_DIV; j++) {
        std::vector<KeyPoint> keyPoints_test;
        Mat dcpt_test;

        Mat roi_image = image2(Rect(i * image2.cols / ROI_U_DIV4, j * image2.rows / ROI_V_DIV, image2.cols / ROI_U_DIV4, image2.rows / ROI_V_DIV));
        detector->detectAndCompute(roi_image, noArray(), keyPoints_test, dcpt_test);

        for (int k = 0; k < keyPoints_test.size(); k++) {
          keyPoints_test[k].pt = Point(i * image1.cols / ROI_U_DIV4 + keyPoints_test[k].pt.x, j * image1.rows / ROI_V_DIV + keyPoints_test[k].pt.y);
          keyPoints_r.push_back(keyPoints_test[k]);
          dcpt_r.push_back(dcpt_test.row(k));
        }
      }
    }

    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    std::vector< std::vector<DMatch> > matches;

    matcher->knnMatch(dcpt_l, dcpt_r, matches, 2);
    //vector<KeyPoint> match1, match2;
    //for (int i = 0; i < matches.size(); i++) {
    //	match1.push_back(keyPoints_l[matches[i][0].queryIdx]);
    //	match2.push_back(keyPoints_r[matches[i][1].trainIdx]);
    //}


    //drawKeypoints(image1, keyPoints_l, image1, Scalar::all(255));
    //drawKeypoints(image2, keyPoints_r, image2, Scalar::all(255));

    Mat res;
    drawMatches(image1, keyPoints_l, image2, keyPoints_r, matches, res);
    line(res, Point(u0_l, 0), Point(u0_l, 479), Scalar(0, 255, 0), 2);
    line(res, Point(u0_r + 640, 0), Point(u0_r + 640, 479), Scalar(0, 255, 0), 2);

    std::vector<KeyPoint> find_landmarks_kp;
    std::vector<Point3d> find_landmarks_o;
    Mat find_landmarks_d;

    find_landmarks_kp.clear();
    find_landmarks_o.clear();
    find_landmarks_d = Mat();

    for (int i = 0; i < matches.size(); i++) {
      if (abs(keyPoints_r[matches[i][0].trainIdx].pt.y - keyPoints_l[matches[i][0].queryIdx].pt.y) < 5) {
        // Compute z
        double Ix_r = keyPoints_r[matches[i][0].trainIdx].pt.x;
        double Iy_r = keyPoints_r[matches[i][0].trainIdx].pt.y;
        double Ix_l = keyPoints_l[matches[i][0].queryIdx].pt.x;
        double Iy_l = keyPoints_l[matches[i][0].queryIdx].pt.y;


        double Hz;
        Hz = L*fu_l;
        Hz /= Ix_l -Ix_r;

        double Hx;
        Hx = L* ((Ix_l - u0_l) + (Ix_r - u0_r));
        Hx /= 2 * (Ix_l - Ix_r);
        Hx += L/2;

        double Hy;
        Hy = L*((Iy_l - v0_l) + (Iy_r - v0_r));
        Hy /= 2 * (Ix_l - Ix_r);

        if (!is_kpinit) {
          map_landmarks_kp.push_back(keyPoints_l[matches[i][0].queryIdx]);
          map_landmarks_o.push_back(Point3d(Hx, Hy, Hz));
          map_landmarks_d.push_back(dcpt_l.row(matches[i][0].queryIdx));
        }
        else {
          find_landmarks_kp.push_back(keyPoints_l[matches[i][0].queryIdx]);
          find_landmarks_o.push_back(Point3d(Hx, Hy, Hz));
          find_landmarks_d.push_back(dcpt_l.row(matches[i][0].queryIdx));
        }


        char buffer[40];
        sprintf(buffer, "(%1.4F, %1.4F, %1.4F)", Hx / 1000.0, Hy / 1000.0, Hz / 1000.0 );


        circle(image1, keyPoints_l[matches[i][0].queryIdx].pt, 5, Scalar(255, 0, 0), 2);
        //putText(image1, buffer, keyPoints_l[matches[i][0].queryIdx].pt, CV_FONT_NORMAL, 0.4, Scalar::all(255), 1);

      }
    }
    //is_sampleMK = false;

    std::vector<Point3d> matched_objectPoints;
    std::vector<Point2d> matched_imagePoints;
    matched_objectPoints.clear();
    matched_imagePoints.clear();

    matches.clear();
    matcher->knnMatch(find_landmarks_d, map_landmarks_d, matches, 2);

    for (int i = 0; i < matches.size(); i++) {
      matched_objectPoints.push_back(map_landmarks_o[matches[i][0].trainIdx]);
      matched_imagePoints.push_back(Point2d(find_landmarks_kp[matches[i][0].queryIdx].pt));

      char buffer[20];
      sprintf(buffer, "%d", matches[i][0].trainIdx);
      putText(image1, buffer, find_landmarks_kp[matches[i][0].queryIdx].pt + Point2f(-20.0,0.0), CV_FONT_NORMAL, 0.6, Scalar(0,0,255), 1.5);
    }
    /*******************  The landmark matching finish  *********************/


    Mat new_matrix = P1(Rect(0, 0, 3, 3));
    Mat map = Mat::zeros(280,280,CV_8UC3);
    if (is_kpinit) {

      // Record current landmarks to map
      map_landmarks_d = find_landmarks_d.clone();
      map_landmarks_kp.swap(find_landmarks_kp);
      map_landmarks_o.swap(find_landmarks_o);

      Mat new_r, new_t, r_matrix;
      solvePnPRansac(matched_objectPoints, matched_imagePoints, new_matrix, Mat(), new_r, new_t,false,500);
      Rodrigues(new_r, r_matrix);

      Mat prediction = KF.predict();

      // Record measure state
      measure.at<float>(0, 0) = new_t.at<double>(0, 0);
      measure.at<float>(1, 0) = new_t.at<double>(1, 0);
      measure.at<float>(2, 0) = new_t.at<double>(2, 0);

//      float alpha = atan2(r_matrix.at<double>(1, 0), r_matrix.at<double>(0, 0));
//      float beta = atan2(
//        -r_matrix.at<double>(2, 0),
//        sqrtf(r_matrix.at<double>(2, 1)*r_matrix.at<double>(2, 1) + r_matrix.at<double>(2, 2)*r_matrix.at<double>(2, 2)));
//      float gamma = atan2(r_matrix.at<double>(2, 1), r_matrix.at<double>(2, 2));

//      measure.at<float>(3, 0) = gamma; // gamma
//      measure.at<float>(4, 0) = beta; // beta
//      measure.at<float>(5, 0) = alpha; // alpha

      // Rotate with X Y Z axis
      Vec3f rpy = rotationMatrixToEulerAngles(r_matrix);
      measure.at<float>(3, 0) = rpy[0]; // row
      measure.at<float>(4, 0) = rpy[1]; // pitch
      measure.at<float>(5, 0) = rpy[2]; // yaw

      KF.correct(measure);

      // Pose_state = [ R(3X3) T(3X1) ]
      //              [    0      1   ]
      Mat new_h_pose = Mat::zeros(4,4,CV_64FC1);
      new_h_pose.at<double>(0,3) = KF.statePost.at<float>(0, 0);
      new_h_pose.at<double>(1,3) = KF.statePost.at<float>(1, 0);
      new_h_pose.at<double>(2,3) = KF.statePost.at<float>(2, 0);
      new_h_pose.at<double>(3,3) = 1.0;

      Vec3f updated_rpy(
            KF.statePost.at<float>(3,0),
            KF.statePost.at<float>(4,0),
            KF.statePost.at<float>(5,0));

      //Mat roi_r_h = new_h_pose(Rect(0,0,3,3));
      Mat roi_r_h = eulerAnglesToRotationMatrix(updated_rpy);
      for(int i=0;i<3;i++)
        for(int j=0;j<3;j++) {
          new_h_pose.at<double>(i,j) = roi_r_h.at<double>(i,j);
        }

      std::cout << "roi_r_h = " << roi_r_h << std::endl;
      pose_state = pose_state * new_h_pose;
      std::cout << "new pose = " << new_h_pose << std::endl;
      std::cout << "Current = " << pose_state << std::endl;
      char buffer[50];
      //sprintf(buffer, "(%4.1f,%4.1f,%4.1f)", new_t.at<double>(0, 0), new_t.at<double>(0, 1), new_t.at<double>(0, 2));
      //sprintf(buffer, "(%5.1f,%5.1f,%5.1f)", KF.statePost.at<float>(0, 0), KF.statePost.at<float>(1, 0), KF.statePost.at<float>(2, 0));
      sprintf(buffer, "(%5.1lf,%5.1lf,%5.1lf)", pose_state.at<double>(0, 3), pose_state.at<double>(1, 3), pose_state.at<double>(2, 3));
      double x = pose_state.at<double>(0, 3);
      double y = pose_state.at<double>(1, 3);
      double z = pose_state.at<double>(2, 3);
      putText(image1, buffer, Point2f(10.0, 20.0), CV_FONT_NORMAL, 0.6, Scalar(0, 255, 255), 1.5);

      for (int j = 0 ; j < map.cols; j += 10) {
        line(map, Point(j, 0), Point(j, map.rows), Scalar::all(50), 1, 8);
      }
      for (int j = 0; j < map.rows; j += 10) {
        line(map, Point(0, j), Point(map.cols, j), Scalar::all(70), 1, 8);
      }
      line(map, Point(map.cols/2, 0), Point(map.cols/2, map.rows), Scalar(128,50,50), 3, 8);
      line(map, Point(0, map.rows/2), Point(map.cols, map.rows/2), Scalar(50,128,50), 3, 8);
      ellipse(map, Point(map.cols/2 + KF.statePost.at<float>(0, 0) / 100, map.cols/2 - KF.statePost.at<float>(2, 0) / 100),
        Size((int)KF.errorCovPost.at<float>(0, 0)*1, (int)KF.errorCovPost.at<float>(2, 2)*1) * 1,0, 0, 360,
        Scalar::all(255), 1);
      //ellipse()

      //(map, Point(320 + KF.statePost.at<float>(0, 0) / 10, 320 - KF.statePost.at<float>(2, 0) / 10), 4, Scalar::all(255), 4);
      //imshow("map", map);
    }
    else {
      for (int j = 0 ; j < map.cols; j += 10) {
        line(map, Point(j, 0), Point(j, map.rows), Scalar::all(50), 1, 8);
      }
      for (int j = 0; j < map.rows; j += 10) {
        line(map, Point(0, j), Point(map.cols, j), Scalar::all(70), 1, 8);
      }
      line(map, Point(map.cols/2, 0), Point(map.cols/2, map.rows), Scalar(128,50,50), 3, 8);
      line(map, Point(0, map.rows/2), Point(map.cols, map.rows/2), Scalar(50,128,50), 3, 8);
    }


    cv::resize(image1, canvasPart1, canvasPart1.size(), 0, 0, INTER_AREA);
    cv::resize(image2, canvasPart2, canvasPart2.size(), 0, 0, INTER_AREA);

    for (int j = 0; j < canvas.rows; j += 16) {
        line(canvas, Point(0, j), Point(canvas.cols, j), Scalar(0, 255, 0), 1, 8);
    }

    cv::resize(res, canvas_raw, canvas_raw.size(), 0, 0, INTER_AREA);

    cvtColor(canvas_raw, canvas_raw, CV_BGR2RGB);
    ui->rawCamView->setPixmap(QPixmap::fromImage
                                   (QImage(reinterpret_cast<const unsigned char*>(canvas_raw.data),
                                    canvas_raw.cols, canvas_raw.rows, QImage::Format_RGB888)));

    cvtColor(canvas, canvas, CV_BGR2RGB);
    ui->rectCamView->setPixmap(QPixmap::fromImage
                                   (QImage(reinterpret_cast<const unsigned char*>(canvas.data),
                                    canvas.cols, canvas.rows, QImage::Format_RGB888)));

    cvtColor(image1, image1, CV_BGR2RGB);
    ui->matchView->setPixmap(QPixmap::fromImage
                                   (QImage(reinterpret_cast<const unsigned char*>(image1.data),
                                    image1.cols, image1.rows, QImage::Format_RGB888)));

    cvtColor(map, map, CV_BGR2RGB);
    ui->poseMap->setPixmap(QPixmap::fromImage
                           (QImage(reinterpret_cast<const unsigned char*>(map.data),
                            map.cols, map.rows, QImage::Format_RGB888)));

    //is_kpinit = true;
}

void MainWindow::on_featureCountSlider_sliderMoved(int position) {
  ui->featureCountNumber->display(position);
  detector->setMaxFeatures(position);
}

void MainWindow::on_checkBox_clicked(bool checked)
{
    is_kpinit = checked;
}

void MainWindow::on_sampleMKButton_clicked()
{
  is_sampleMK = true;
}

