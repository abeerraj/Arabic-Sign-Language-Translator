#import <opencv2/opencv.hpp>
#import <opencv2/highgui/cap_ios.h>
#import <opencv2/highgui/ios.h>
#import <opencv2/features2d/features2d.hpp>
#import <opencv2/nonfree/nonfree.hpp>
#include <opencv2/ml/ml.hpp>
#import <opencv2/nonfree/features2d.hpp>
#include <sqlite3.h>
#include "OpenCVWrapper.h"

using namespace cv;
using namespace std;
NSString* const faceCascadeFilename = @"haarcascade_frontalface_alt";
const int HaarOptions = CV_HAAR_FIND_BIGGEST_OBJECT | CV_HAAR_DO_ROUGH_SEARCH;

@interface
OpenCVWrapper () <CvVideoCameraDelegate>
@end

@implementation OpenCVWrapper
{
    UIViewController<CvCameraDelegate> * delegate;
    UIImageView * imageView;
    CvVideoCamera * videoCamera;
    int count;
    Mat bgImage;
    CvSVM svm;
    cv::Mat vocabulary;
    CascadeClassifier faceCascade;
    Mat originalFrame;
    sqlite3 *db;
    int rc; //row count for db
    int preIndex;//Check pre translation
    NSString* word;
    string dbWord;
}

//return word to swift
- (NSString*) returnstring
{
    return word;
}

- (void)getString:(int)i
{
    //check pre translation
    if (i == -1 || preIndex == i)
    {
        return;
    }
    string sqlStr ="SELECT Word FROM ArSLT WHERE id = ";
    sqlStr += to_string(i);
    
    const char * sql = sqlStr.c_str();
    sqlite3_stmt *stmt = NULL;
    
    rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK)
        cout << "something wrong";
    rc = sqlite3_step(stmt);
    
    while (rc != SQLITE_DONE && rc != SQLITE_OK)
    {
        int colCount = sqlite3_column_count(stmt);
        for (int colIndex = 0; colIndex < colCount; colIndex++)
        {
            string columnName = sqlite3_column_name(stmt, colIndex);
            if (columnName == "Word")
            {
                dbWord = std::string(reinterpret_cast<const char*>(sqlite3_column_text(stmt, colIndex)));
            }
        }
        rc = sqlite3_step(stmt);
    }
    rc = sqlite3_finalize(stmt);
    preIndex = i;
}

//create training data
- (void) createTrainDataUsingBow:(cv::Mat&)train andresponse:(cv::Mat&)response
{
    NSString* path;
    cv::Mat image;
    int numWords = 2;
    int sample = 60;
    int cluster= 200;
    
    cv::initModule_nonfree(); //initalizes SURF
    cv::Ptr<cv::DescriptorMatcher> matcher = new FlannBasedMatcher;
    cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SurfDescriptorExtractor();
    cv::BOWImgDescriptorExtractor dextract( extractor, matcher ); //Bag of Words
    cv::SurfFeatureDetector detector(400,4,2);
    detector.extended = true;
    cv::BOWKMeansTrainer bow( cluster,  cv::TermCriteria(CV_TERMCRIT_ITER,100,0.001), 1, cv::KMEANS_PP_CENTERS );

    //read image by image
    for (int j = 1 ; j <= numWords; j++)
    {
        for(int i = 1; i <= sample; i++)
        {
            path = [NSString stringWithFormat:@"image%d_%d.jpg",j,i];
            UIImage* img1 = [UIImage imageNamed:path];
            UIImageToMat(img1, image);
            
            //get image keypoints, descriptors
            std::vector<cv::KeyPoint> keypoints;
            cv::Mat descriptors;
            detector.detect( image, keypoints);
            extractor->compute( image, keypoints, descriptors);
            if (!descriptors.empty())
            {
                bow.add(descriptors);
            }
        }
    }
    
    // Create the vocabulary with KMeans.
    vocabulary = bow.cluster();
    dextract.setVocabulary(vocabulary);
    
    for (int j = 1 ; j <= numWords; j++)
    {
        for(int i = 1; i <= sample; i++)
        {
            path = [NSString stringWithFormat:@"image%d_%d.jpg",j,i];
            UIImage* img1 = [UIImage imageNamed:path];
            UIImageToMat(img1, image);
            std::vector<cv::KeyPoint> keypoints;
            detector.detect( image, keypoints);
            cv::Mat desc;
            
            dextract.compute(image, keypoints, desc);
            if (!desc.empty())
            {
                train.push_back(desc);            // update training data
                response.push_back((float)j);     // update response data
            }
        }
    }
}

//train data using svm
-(int) trainSVM
{
    Mat response;
    Mat train;
    [self createTrainDataUsingBow:train andresponse:response];
    
    //train svm
    svm.train_auto(train, response, cv::Mat(), cv::Mat(), svm.get_params());
    
    return 0;
}


//Initalize the camera
- (id)initWithController:(UIViewController<CvCameraDelegate>*)c andImageView:(UIImageView*)iv
{
    //1.train svm
    [self trainSVM];
    
    //initialize camera
    delegate = c;
    imageView = iv;
    videoCamera = [[CvVideoCamera alloc] initWithParentView:imageView];
    self->videoCamera.defaultAVCaptureVideoOrientation =  AVCaptureVideoOrientationPortrait; //Default orientation is portait
    videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack; //Intially back camera is opened
    videoCamera.defaultFPS = 20; // How often 'processImage' is called, (30 frames per second)
    videoCamera.delegate = self;
    count = 0;
    NSString* faceCascadePath = [[NSBundle mainBundle] pathForResource:faceCascadeFilename ofType:@"xml"]; //trained data set from OpenCV used for face detection
    faceCascade.load([faceCascadePath UTF8String]);
    
    //open sqlite
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *documentsDirectory = [paths objectAtIndex:0];
    NSString *documentTXTPath = [documentsDirectory stringByAppendingPathComponent:@"ArSLT.sqlite"];
    const char *fileName = [documentTXTPath cStringUsingEncoding:NSASCIIStringEncoding];
    if (sqlite3_open( fileName , &db) != SQLITE_OK)
    {
        cout << "Could not open database";
    }
    preIndex = 0;
    return self;
}
-(bool)run
{
    return [self->videoCamera running];
}

//Processing frame
- (void)processImage:(cv::Mat &)img
{
    Mat binaryImg;
    int index;
    originalFrame = img; //display original frame
    
    binaryImg = [self handSegmentation:img];
    index = [self handFeatureExtractionandGestureMatching:binaryImg];
    [self getString:index];
    
    NSString* result = [[NSString alloc] initWithUTF8String:dbWord.c_str()];
    word = result;
}

- (cv::Mat)handSegmentation:(cv::Mat &)img
{
    Mat ycrcb;
    
    //1. get background frame
    if (count == 30)
    {
        img.copyTo(bgImage);
        
        //convert background image to YCrCb
        cv::cvtColor(bgImage,bgImage,COLOR_BGR2YCrCb);
        count++;
    }
    else if (count < 30)
        count++;
    
    //2. Convert incoming img to ycrcb to match template
    cv::cvtColor(img, ycrcb, COLOR_BGR2YCrCb);
    
    //3. Background subtraction
    Mat diffImage = abs(ycrcb-bgImage);
    
    //4. Split image into 3 channels
    Mat chan[3];
    split(diffImage,chan);
    Mat y = chan[0];
    Mat Cr = chan[1];
    Mat Cb = chan[2];
    
    //5. Thresholding and Binarization
    threshold(y, y, 54, 255, THRESH_BINARY);
    threshold(Cr, Cr, 131, 255, THRESH_BINARY);
    threshold(Cb, Cb, 110, 255, THRESH_BINARY);
    
    //6. Morphology
    int elementSize = 2;
    Mat kernel = getStructuringElement(MORPH_RECT, cv::Size(2 * elementSize + 1, 2 * elementSize + 1), cv::Point(elementSize,elementSize));
    morphologyEx(y, y, MORPH_OPEN, kernel);
    morphologyEx(Cr, Cr, MORPH_OPEN, kernel);
    morphologyEx(Cb, Cb, MORPH_OPEN, kernel);
    
    //7. addition
    diffImage = y + Cr + Cb;
    
    //8. morphology for binary image
    elementSize = 3;
    kernel = getStructuringElement(MORPH_ELLIPSE, cv::Size(2 * elementSize + 1, 2 * elementSize + 1), cv::Point(elementSize,elementSize));
    morphologyEx(diffImage, diffImage, MORPH_CLOSE, kernel);
    morphologyEx(diffImage, diffImage, MORPH_CLOSE, kernel);
    
    //9. (ycrcb + diffImage)
    ycrcb.copyTo(ycrcb, diffImage);
    
    //10. face detection
    Mat grayscaleFrame;
    cvtColor(originalFrame, grayscaleFrame, CV_BGR2GRAY);
    equalizeHist(grayscaleFrame, grayscaleFrame);
    std::vector<cv::Rect> faces;
    faceCascade.detectMultiScale(grayscaleFrame, faces, 1.1, 2, HaarOptions, cv::Size(60, 60));
    
    bool flag = true;
    for (int i = 0; i < faces.size(); i++)
    {
        //11. face removal
        cv::Point center=cv::Point( faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5 );
        cv::circle( ycrcb, center,faces[i].width/1.5,cvScalar( 255, 0, 255 ), CV_FILLED, 8, 0 );
        flag = false;
    }
    
    //12. Skin extraction
    Mat binaryImg;
    inRange(ycrcb, Scalar(54, 127, 77), Scalar(163, 173, 127), binaryImg);
    
    //13. Morphology
    morphologyEx(binaryImg, binaryImg, MORPH_OPEN, kernel);
    
    //14. Gaussian Filter
    GaussianBlur(binaryImg, binaryImg, cv::Size(5,5), 0);
    
    return binaryImg;
}

- (int)handFeatureExtractionandGestureMatching:(cv::Mat &)binaryImg
{
    cv::Mat desc_bow;
    vector<cv::KeyPoint> keypoints;
    cv::initModule_nonfree();
    cv::Ptr<cv::DescriptorMatcher> matcher =new FlannBasedMatcher;
    cv::Ptr<cv::DescriptorExtractor> extractor = new cv::SurfDescriptorExtractor();
    cv::BOWImgDescriptorExtractor dextract( extractor, matcher );
    dextract.setVocabulary(vocabulary);
    cv::SurfFeatureDetector detector(400,4,2);
    detector.extended = true;
    
    detector.detect( binaryImg, keypoints);
    dextract.compute( binaryImg, keypoints, desc_bow );

    float res;
    if(!desc_bow.empty()){
        res = svm.predict(desc_bow);
    }
    else{
        res = -1;
    }
    
    return res;
}

//Change Camera Position
- (void)switchCamera
{
    [self->videoCamera switchCameras];
}

//Start the camera
- (void)start
{
    [videoCamera start];
}

//Stops the camera
- (void)stop
{
    count = 0;
    if([videoCamera running])
    {
        [videoCamera stop];
    }
}
@end
