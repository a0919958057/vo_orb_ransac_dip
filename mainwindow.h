#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/opencv.hpp>

#include <QTimer>
using namespace cv;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;
    QTimer* timer;
    Mat image1;
    Mat image2;

    Mat canvas_raw;
    Mat canvas;

    VideoCapture cap_left;
    VideoCapture cap_right;

    Mat coff_k1, coff_k2, coff_d1, coff_d2, map_l_1, map_l_2, map_r_1, map_r_2;
    Mat r,t,e,f, R1, R2, P1, P2, Q;

    // Kalman filter
    KalmanFilter KF;
    Mat measure;

    // Pose state matrix
    Mat pose_state;

    float u0_l;
    float u0_r;
    //float u0 = 340;
    float v0_l;
    float v0_r;

    float fu_l;
    float fu_r;

    float fv_l;
    float fv_r;

    Ptr<ORB> detector;

    float L;

    std::vector<KeyPoint> map_landmarks_kp;
    std::vector<Point3d>	map_landmarks_o;
    Mat map_landmarks_d;

    bool is_kpinit = false;
    bool is_sampleMK = false;

    long id_counter;

    void calib_steo();

private slots:
    void showImage();
    void on_featureCountSlider_sliderMoved(int position);
    void on_checkBox_clicked(bool checked);
    void on_sampleMKButton_clicked();
};

#endif // MAINWINDOW_H
