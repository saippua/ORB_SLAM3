#include <iostream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <ostream>
#include <sstream>
#include <unistd.h>

#include <opencv2/core/core.hpp>

#include <System.h>
#include "ImuTypes.h"
#include "Optimizer.h"

#include <filesystem> // Requires C++17


using namespace std;

void LoadImages(const string &strPathLeft, const string &strPathRight,
        const std::filesystem::directory_iterator &strPathTimes, 
        vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps);

void LoadIMU(const string &strAccelPath, const string &strGyroPath, 
        vector<double> &vTimestampsImu, 
        vector<cv::Point3f> &vAccel, vector<cv::Point3f> &vGyro);

cv::Point3f splitCSV(string row, const char delimiter, double& stamp);

cv::Point3f interpolateMeasure(const double target_time,
        const cv::Point3f current_data, const double current_time,
        const cv::Point3f prev_data, const double prev_time);

int main(int argc, char **argv)
{
    if(argc < 4)
    {
        cerr << endl << "Usage: ./stereo_inertial_simod";
        cerr << " path_to_vocabulary";
        cerr << " path_to_settings"; 
        cerr << " path_to_dataset";
        cerr << " (path_to_dataset_2 ... path_to_dataset_N) " << endl;
        return 1;
    }

    const int num_seq = argc-3;
    cout << "num_seq = " << num_seq << endl;
    bool bFileName = ((argc-3) == 1);
    string file_name;
    if (bFileName)
    {
        file_name = string(argv[argc-1]);
        cout << "file name: " << file_name << endl;
    }


    // Load all sequences:
    int seq;
    vector< vector<string> > vstrImageLeft;
    vector< vector<string> > vstrImageRight;
    vector< vector<double> > vTimestampsCam;
    vector< vector<cv::Point3f> > vAccel, vGyro;
    vector< vector<double> > vTimestampsImu;

    vector<int> nImages;
    vector<int> nImu;
    vector<int> first_imu(num_seq,0);

    vstrImageLeft.resize(num_seq);
    vstrImageRight.resize(num_seq);
    vTimestampsCam.resize(num_seq);
    vAccel.resize(num_seq);
    vGyro.resize(num_seq);
    vTimestampsImu.resize(num_seq);
    nImages.resize(num_seq);
    nImu.resize(num_seq);

    int tot_images = 0;
    for (seq = 0; seq<num_seq; seq++)
    {
        cout << "Loading images for sequence " << seq << "...";

        string pathSeq(argv[seq + 3]);
        auto imageTimeStamps = std::filesystem::directory_iterator(pathSeq + "/cam0/raw");
    
        string pathCam0 = pathSeq + "/cam0/raw/";
        string pathCam1 = pathSeq + "/cam1/raw/";
        string pathAccel = pathSeq + "/accel/accel_data.csv";
        string pathGyro = pathSeq + "/gyro/gyro_data.csv";
        

        LoadImages(pathCam0, pathCam1, imageTimeStamps, vstrImageLeft[seq], vstrImageRight[seq], vTimestampsCam[seq]);
        cout << "LOADED!" << endl;

        cout << "Loading IMU for sequence " << seq << "...";
        LoadIMU(pathAccel, pathGyro, vTimestampsImu[seq], vAccel[seq], vGyro[seq]);
        cout << "LOADED!" << endl;

        nImages[seq] = vstrImageLeft[seq].size();
        tot_images += nImages[seq];
        nImu[seq] = vTimestampsImu[seq].size();

        if((nImages[seq]<=0)||(nImu[seq]<=0))
        {
            cerr << "ERROR: Failed to load images or IMU for sequence" << seq << endl;
            return 1;
        }

        while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][0])
            first_imu[seq]++;
        first_imu[seq]--; // first imu measurement to be considered
    }

    cout << "Loading settings.. ";
    cv::FileStorage fsSettings(argv[2], cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }
    cout << "Done!" << endl;

    // Vector for tracking time statistics
    vector<float> vTimesTrack;
    vTimesTrack.resize(tot_images);

    cout << endl << "-------" << endl;
    // cout.precision(17);

    ORB_SLAM3::System SLAM(argv[1],argv[2],ORB_SLAM3::System::IMU_STEREO, false);

    std::ifstream fp;
    cv::Mat im_left(800, 848, CV_8UC1);
    cv::Mat im_right(800, 848, CV_8UC1);
    for (seq = 0; seq<num_seq; seq++)
    {
        // Seq loop
        vector<ORB_SLAM3::IMU::Point> vImuMeas;
        double t_rect = 0.f;
        double t_resize = 0.f;
        double t_track = 0.f;
        int num_rect = 0;
        int proccIm = 0;
        for(int ni=0; ni<nImages[seq]; ni++, proccIm++)
        {
            // Read left and right images from file
            fp.open(vstrImageLeft[seq][ni].c_str(), std::ios::binary);
            fp.read((char*)im_left.data, im_left.total());
            fp.close();

            fp.open(vstrImageRight[seq][ni].c_str(), std::ios::binary);
            fp.read((char*)im_right.data, im_right.total());
            fp.close();
            
            // im_left = cv::imread(vstrImageLeft[seq][ni],cv::IMREAD_GRAYSCALE);
            // im_right = cv::imread(vstrImageRight[seq][ni],cv::IMREAD_GRAYSCALE);


            if(im_left.empty())
            {
                cerr << endl << "Failed to load image at: "
                     << string(vstrImageLeft[seq][ni]) << endl;
                return 1;
            }

            if(im_right.empty())
            {
                cerr << endl << "Failed to load image at: "
                     << string(vstrImageRight[seq][ni]) << endl;
                return 1;
            }

            double tframe = vTimestampsCam[seq][ni];

            cout << "Time: " << tframe << endl;
            // Load imu measurements from previous frame
            vImuMeas.clear();

            if(ni>0) {
                while(vTimestampsImu[seq][first_imu[seq]]<=vTimestampsCam[seq][ni])
                {
                    vImuMeas.push_back(ORB_SLAM3::IMU::Point(
                        vAccel[seq][first_imu[seq]].x,vAccel[seq][first_imu[seq]].y,vAccel[seq][first_imu[seq]].z,
                        vGyro[seq][first_imu[seq]].x,vGyro[seq][first_imu[seq]].y,vGyro[seq][first_imu[seq]].z,
                        vTimestampsImu[seq][first_imu[seq]]));
                    first_imu[seq]++;
                }
            }

                

    #if defined(COMPILEDWITHC11) || defined(COMPILEDWITHC17)
            std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();
    #endif

            // Pass the images to the SLAM system
            SLAM.TrackStereo(im_left,im_right,tframe,vImuMeas);

    #if defined(COMPILEDWITHC11) || defined(COMPILEDWITHC17)
            std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
    #else
            std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
    #endif

#ifdef REGISTER_TIMES
            t_track = t_rect + t_resize + std::chrono::duration_cast<std::chrono::duration<double,std::milli> >(t2 - t1).count();
            SLAM.InsertTrackTime(t_track);
#endif

            double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();

            vTimesTrack[ni]=ttrack;

            // Wait to load the next frame
            double T=0;
            if(ni<nImages[seq]-1)
                T = vTimestampsCam[seq][ni+1]-tframe;
            else if(ni>0)
                T = tframe-vTimestampsCam[seq][ni-1];

            if(ttrack<T)
                usleep((T-ttrack)*1e6); // 1e6
        }

        if(seq < num_seq - 1)
        {
            cout << "Changing the dataset" << endl;

            SLAM.ChangeDataset();
        }
    }
    SLAM.Shutdown();

    std::filesystem::create_directories(file_name + "/orb_slam");
    SLAM.SaveTrajectoryEuRoC(file_name + "/orb_slam/CameraTrajectory.csv");
    SLAM.SaveKeyFrameTrajectoryEuRoC(file_name + "/orb_slam/KeyFrameTrajectory.csv");

    return 0;
}

void LoadImages(const string &strPathLeft, const string &strPathRight, 
        const std::filesystem::directory_iterator &strPathTimes, 
        vector<string> &vstrImageLeft, vector<string> &vstrImageRight, vector<double> &vTimeStamps)
{
    // ifstream fTimes;
    // fTimes.open(strPathTimes.c_str());
    vTimeStamps.reserve(5000);
    vstrImageLeft.reserve(5000);
    vstrImageRight.reserve(5000);
    for (auto const& entry : strPathTimes)
    {
        string s;
        if (entry.path().extension() != ".raw")
            continue;
        s = entry.path().stem();
        // getline(fTimes,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".raw");
            vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".raw");
            double t;
            ss >> t;
            vTimeStamps.push_back(t/1e9);
        }
    }
    // Sort images in ascending order by timestamp
    sort(vstrImageLeft.begin(), vstrImageLeft.end());
    sort(vstrImageRight.begin(), vstrImageRight.end());
    sort(vTimeStamps.begin(), vTimeStamps.end());
}

void LoadIMU(const string &strAccelPath, const string &strGyroPath, 
        vector<double> &vTimestampsImu, 
        vector<cv::Point3f> &vAccel, vector<cv::Point3f> &vGyro)
{
    // Open csv files and skip header
    ifstream fAccel, fGyro;
    fAccel.open(strAccelPath.c_str());
    fAccel.ignore(numeric_limits<streamsize>::max(), '\n');
    fGyro.open(strGyroPath.c_str());
    fGyro.ignore(numeric_limits<streamsize>::max(), '\n');

    vTimestampsImu.reserve(5000);
    vAccel.reserve(5000);
    vGyro.reserve(5000);


    // Get starting value
    string sg0, sa0, sa1;
    getline(fGyro, sg0);
    getline(fAccel, sa0);
    getline(fAccel, sa1);

    // Initialize variables for holding IMU data
    double prev_accel_timestamp;
    double current_accel_timestamp;
    double current_gyro_timestamp;
    cv::Point3f prev_accel_data = splitCSV(sa0, ' ', prev_accel_timestamp);
    cv::Point3f current_accel_data = splitCSV(sa1, ' ', current_accel_timestamp);
    cv::Point3f current_gyro_data = splitCSV(sg0, ' ', current_gyro_timestamp);

    vTimestampsImu.push_back(current_gyro_timestamp/1e3);
    vGyro.push_back(current_gyro_data);

    // This loop loads gyro and accel data concurrently. Gyro data is in 200Hz and Accel data is 62.5Hz.
    // Each time a gyro datapoint is loaded, an accel datapoint is interpolated at the timestamp using prev_accel_data and current_accel_data.
    // If gyro timestamp catches up to the accel timestamp, a new accel datapoint is loaded from the CSV.
    while (!fGyro.eof() && !fAccel.eof()) {
        string s;
        if (current_accel_timestamp > current_gyro_timestamp) {
            cv::Point3f interp_data = interpolateMeasure(current_gyro_timestamp, current_accel_data, current_accel_timestamp, prev_accel_data, prev_accel_timestamp);
            vAccel.push_back(interp_data);

            // Get new gyroscope datapoint
            getline(fGyro, s);
            if (s.size() == 0) continue; // Ignore empty lines
            current_gyro_data = splitCSV(s, ' ', current_gyro_timestamp);
            vTimestampsImu.push_back(current_gyro_timestamp/1e3);
            vGyro.push_back(current_gyro_data);
        } else {
            getline(fAccel, s);
            if (s.size() == 0) continue; // Ignore empty lines
            prev_accel_timestamp = current_accel_timestamp;
            prev_accel_data = current_accel_data;
            current_accel_data = splitCSV(s, ' ', current_accel_timestamp);
        }
    }
}

/**
 * Takes a CSV row as a string and splits it into a Point3f vector.
 * The timestamp of the CSV row (first element) is stored in the stamp argument
 */
cv::Point3f splitCSV(string row, const char delimiter, double& stamp) {
    string item;
    size_t pos = row.find(' ');
    double data[3];
    int count=0;

    item = row.substr(0, pos);
    stamp = stod(item);
    row.erase(0, pos+1);
    
    while ((pos = row.find(delimiter)) != string::npos) {
        item = row.substr(0, pos);
        data[count++] = stod(item);
        row.erase(0, pos+1);
    }
    if (count < 3) {
        item = row.substr(0, pos);
        data[count] = stod(item);
    }
    return cv::Point3f(data[0], data[1], data[2]);
}

/**
 * Interpolate a new datapoint between two Point3f vectors at a new timestamp between the two given point's timestamps.
 */
cv::Point3f interpolateMeasure(const double target_time,
        const cv::Point3f current_data, const double current_time,
        const cv::Point3f prev_data, const double prev_time) {

    // If there are not previous information, the current data is propagated
    if(prev_time == 0){
        return current_data;
    }

    cv::Point3f increment;
    cv::Point3f value_interp;

    if(target_time > current_time) {
        value_interp = current_data;
    }
    else if(target_time > prev_time){
        increment.x = current_data.x - prev_data.x;
        increment.y = current_data.y - prev_data.y;
        increment.z = current_data.z - prev_data.z;

        double factor = (target_time - prev_time) / (current_time - prev_time);

        value_interp.x = prev_data.x + increment.x * factor;
        value_interp.y = prev_data.y + increment.y * factor;
        value_interp.z = prev_data.z + increment.z * factor;

        // zero interpolation
        // value_interp = current_data;
    }
    else {
        value_interp = prev_data;
    }

    return value_interp;
}
