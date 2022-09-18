#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include "opencv2/imgproc.hpp"
#include<iomanip>
#include<cmath>
#include "spline.h"
#include "Interpolator.h"
#include "SGSmooth.hpp"
using namespace cv;
using namespace std;

vector<double> CalculateDFT(vector<double>& input, int length, int flag){
    Mat padded;
    Mat converted(length, 1, CV_32F);
    vector<double> output={};

    if(input.size()<length){
        for (int i = input.size(); i < length; ++i) {
            input.push_back(0);
        }
    }
    for (int i = 0; i < length; ++i) {
        converted.at<float>(i,0)=input[i];
    }
    int m = getOptimalDFTSize( converted.rows );
    int n = getOptimalDFTSize( converted.cols ); // on the border add zero values
    copyMakeBorder(converted, padded, 0, m - converted.rows, 0, n - converted.cols, BORDER_CONSTANT, Scalar::all(0));

    Mat planes[] = {Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);         // Add to the expanded another plane with zeros
    dft(complexI, complexI ,flag);            // this way the result may fit in the source matrix

    for (int i = 0; i < complexI.rows; ++i) {
        for (int j = 0; j < complexI.cols; ++j) {
           output.push_back(abs(complexI.at<Vec2f>(i,j)[0]));
        }
    }
    return output;
}

vector<double> FftShift(vector<double> input) {
    int center = (int) floor((float)input.size()/2);
    if (input.size() % 2 != 0) {
        center++;
    }
    vector<double> output={};
    for (int i = 0; i < center; ++i) {
        output.push_back(input.at(input.size() / 2 + i));
    }
    for (int i = 0; i < center; ++i) {
        output.push_back(input.at(i));
    }
    return output;
}

tuple<vector<int>, vector<int>>findNAN(vector<double>& array)
{
    vector<int> nanValues;
    vector<int> nonNAnValues;
    for (int i = 0; i < array.size(); ++i) {
        if (isnan(array.at(i))) {
            nanValues.push_back(i);
        }
        else {
            nonNAnValues.push_back(i);
        }
    }
    //return nanValues and nonNanValues
    return make_tuple(nanValues, nonNAnValues);
}
vector<double> NanHelper(vector<double>& binMean)
{
    vector<int> nanValues;
    vector<int> nonNanValues;
    vector<double> nonNanBinMean;
    tie(nanValues, nonNanValues) = findNAN(binMean);

    for (int i = 0; i < nonNanValues.size(); ++i) {
        nonNanBinMean.push_back(binMean.at(nonNanValues.at(i)));
    }
    vector<pair<double, double>>tempPair;
    for (int i = 0; i < nonNanBinMean.size(); ++i) {
        tempPair.push_back(make_pair(nonNanValues.at(i), nonNanBinMean.at(i)));
    }
    Interpolator interp1{
            {
                    tempPair
            }
    };
    for (int i = 0; i < nanValues.size(); ++i) {
        binMean.at(nanValues.at(i)) = interp1.findValue(nanValues.at(i));
    }
    return binMean;
}

double * secondFit(vector<Point> vector){
    int i,j,k,n,N;
    double static *returnedArray = new double[2];
    //set precision
    cout.precision(4);
    cout.setf(ios::fixed);

    //To find the size of arrays that will store x,y, and z values
    N=vector.capacity();

    double x[N],y[N];
    for(i=0;i<N;i++)
    {
        x[i]=vector[i].x;
        y[i]=vector[i].y;
    }

    // n is the degree of Polynomial
    n=1;

    //Array that will store the values of sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    double X[2*n+1];
    for (i=0;i<2*n+1;i++)
    {
        X[i]=0;
        for (j=0;j<N;j++)
            X[i]=X[i]+pow(x[j],i);        //consecutive positions of the array will store N,sigma(xi),sigma(xi^2),sigma(xi^3)....sigma(xi^2n)
    }
    double B[n+1][n+2],a[n+1];            //B is the Normal matrix(augmented) that will store the equations, 'a' is for value of the final coefficients
    for (i=0;i<=n;i++)
        for (j=0;j<=n;j++)
            B[i][j]=X[i+j];            //Build the Normal matrix by storing the corresponding coefficients at the right positions except the last column of the matrix
    double Y[n+1];                    //Array to store the values of sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    for (i=0;i<n+1;i++)
    {
        Y[i]=0;
        for (j=0;j<N;j++)
            Y[i]=Y[i]+pow(x[j],i)*y[j];        //consecutive positions will store sigma(yi),sigma(xi*yi),sigma(xi^2*yi)...sigma(xi^n*yi)
    }
    for (i=0;i<=n;i++)
        B[i][n+1]=Y[i];                //load the values of Y as the last column of B(Normal Matrix but augmented)
    n=n+1;                //n is made n+1 because the Gaussian Elimination part below was for n equations, but here n is the degree of polynomial and for n degree we get n+1 equations

    for (i=0;i<n;i++)                    //From now Gaussian Elimination starts(can be ignored) to solve the set of linear equations (Pivotisation)
        for (k=i+1;k<n;k++)
            if (B[i][i]<B[k][i])
                for (j=0;j<=n;j++)
                {
                    double temp=B[i][j];
                    B[i][j]=B[k][j];
                    B[k][j]=temp;
                }

    for (i=0;i<n-1;i++)            //loop to perform the gauss elimination
        for (k=i+1;k<n;k++)
        {
            double t=B[k][i]/B[i][i];
            for (j=0;j<=n;j++)
                B[k][j]=B[k][j]-t*B[i][j];    //make the elements below the pivot elements equal to zero or elimnate the variables
        }
    for (i=n-1;i>=0;i--)                //back-substitution
    {                        //x is an array whose values correspond to the values of x,y,z..
        a[i]=B[i][n];                //make the variable to be calculated equal to the rhs of the last equation
        for (j=0;j<n;j++)
            if (j!=i)            //then subtract all the lhs values except the coefficient of the variable whose value
                a[i]=a[i]-B[i][j]*a[j];
        a[i]=a[i]/B[i][i];            //now finally divide the rhs by the coefficient of the variable to be calculated
    }
    for (i=0;i<n;i++)
    {
        returnedArray[i] = a[i];
    }

    return returnedArray;
}

Mat ROISelection(String filePath) {
    Mat img = imread(filePath, IMREAD_COLOR);
    if (img.empty()){
        printf("Image not found\n");
    }
    //Print the height and width of the image
    cout << "Original Image Size : " << img.size() << endl;
    cout << "Width : " << img.size().width << endl;
    cout << "Height: " << img.size().height << endl;
    cout << "Channels: " << img.channels() << endl;

    int start_x = 100;
    int end_x = 250;
    int start_y = 100;
    int end_y = 200;
    int roiArray[4] = {start_y, end_y,start_x, end_x};

    // Slicing to crop the image
    Mat crop = img(Range(start_x,end_x),Range(start_y,end_y));

    //Print the height and width of the cropped image
    cout << "Cropped Image Size : " << crop.size() << endl;
    cout << "Width : " << crop.size().width << endl;
    cout << "Height: " << crop.size().height << endl;
    cout << "Channels: " << crop.channels() << endl;

    return crop;
}

class EventHandler {
public:
    void filePath(String path);
    void LineSelectCallBack();
    void EventExitManager();
};
void EventHandler::filePath(String path){
    cout << path << endl;
}
void EventHandler::LineSelectCallBack() {
    cout << "LineSelectCallBack" << endl;
}
void EventHandler::EventExitManager() {
    cout << "EventExitManager" << endl;
}

class PDSComputeMTF{
    Mat image;
    vector<double> selfEsf;
    vector<double> smoothEsf;
    vector<double> selfLsf;
    vector<double> smoothLsf;
    vector<double> selfMtf;
public:
    void Init(Mat &img);
    void computeESF(double d);
    void computeLSF();
    void computeMTF();
};

void PDSComputeMTF::Init(Mat &img) {
    Mat imgOriginal;        // input image
    Mat imgGrayscale;       // grayscale of input image
    Mat imgBlurred;         // intermediate blurred image
    Mat imgCanny;           // Canny edge image
    //int roiArray[4] = {100, 200, 100, 250};

    imgOriginal = img;
    image = imgOriginal;

    Mat selfData = image;
    cvtColor(selfData, selfData, COLOR_BGR2GRAY ,1);
    Mat destination;
    Mat selfThreshold;
    Mat belowThreshold;
    Mat aboveThreshold;
    double areaBelowThreshold;
    double areaAboveThreshold;

    threshold( selfData, destination, 0, 255, THRESH_BINARY + THRESH_OTSU );
    double min, max;
    minMaxLoc(destination, &min, &max);
    selfThreshold= (destination / (max-min)) + min;

    //convert to double
    selfData.convertTo(selfData, CV_64F);
    selfThreshold.convertTo(selfThreshold, CV_64F);
    //set size of the belowThreshold and aboveThreshold
    belowThreshold = Mat::zeros(selfData.size(), selfData.type());
    aboveThreshold = Mat::zeros(selfData.size(), selfData.type());
    double sumBelowThreshold = 0;
    double sumAboveThreshold = 0;
    double sumSelfData = 0;
    Mat sumMatrixBelowThreshold = Mat::zeros(selfData.size(), selfData.type());
    Mat sumMatrixAboveThreshold = Mat::zeros(selfData.size(), selfData.type());
    sumMatrixAboveThreshold.convertTo(sumMatrixAboveThreshold, CV_64F);
    sumMatrixBelowThreshold.convertTo(sumMatrixBelowThreshold, CV_64F);

    for (int i = 0; i < selfData.rows; i++) {
        for (int j = 0; j < selfData.cols; j++) {
            if(selfData.at<double>(i,j) >= min && selfData.at<double>(i,j) <= selfThreshold.at<double>(i,j)) {
                belowThreshold.at<double>(i,j) = selfData.at<double>(i,j);
                sumBelowThreshold += 1;
                sumMatrixBelowThreshold.at<double>(i,j) = selfData.at<double>(i,j);
                //sumBelowThreshold * selfData.rows;
            }
            if(selfData.at<double>(i,j) >= selfThreshold.at<double>(i,j) && selfData.at<double>(i,j) <= max) {
                aboveThreshold.at<double>(i,j) = selfData.at<double>(i,j);
                sumAboveThreshold += 1;
                sumMatrixAboveThreshold.at<double>(i,j) = selfData.at<double>(i,j);
            }
            sumSelfData += selfData.at<double>(i,j);
        }
    }
    //find sum sumMatrixBelowThreshold
    double sumBelowThresholdMatrix = 0;
    double sumAboveThresholdMatrix = 0;
    double selfThresholdResult = 0;
    for (int i = 0; i < sumMatrixBelowThreshold.rows; i++) {
        for (int j = 0; j < sumMatrixBelowThreshold.cols; j++) {
            sumBelowThresholdMatrix += sumMatrixBelowThreshold.at<double>(i,j);
            sumAboveThresholdMatrix += sumMatrixAboveThreshold.at<double>(i,j);
        }
    }
    areaBelowThreshold = (sumBelowThresholdMatrix ) / sumBelowThreshold;
    areaAboveThreshold = (sumAboveThresholdMatrix ) / sumAboveThreshold;
    selfThresholdResult=(areaBelowThreshold - areaAboveThreshold) / 2+ areaAboveThreshold;

    // convert to grayscale
    cvtColor(imgOriginal, imgGrayscale, COLOR_BGR2GRAY ,1);

    image=imgGrayscale;

    // blur the image to reduce noise
    GaussianBlur(imgGrayscale,                          // input image
                     imgBlurred,                        // output image
                     cv::Size(5, 5),      // smoothing window width and height in pixels
                     1.5);                           // sigma value, determines how much the image will be blurred

    // canny edge detection
    Canny(imgBlurred,            // input image
             imgCanny,           // output image
             0,             // low threshold
             250);            // high threshold

    //finding all white pixels
    vector<Point> whitePixels;
    for (int i = 0; i < imgCanny.rows; i++) {
        for (int j = 0; j < imgCanny.cols; j++) {
            if (imgCanny.at<uchar>(i, j) == 255) {
                whitePixels.push_back(Point(j, i));
            }
        }
    }
    double *array = {};
    array = secondFit(whitePixels);   //array[0] --> intercept    //array[1] --> slope

    double angleRadians = atan(array[1]);
    double angleDegree = angleRadians * 180 / (3.14);
    angleDegree = abs(angleDegree);
    if(angleDegree < 45) {
        //transpose the image
        Mat transposed_image;                           //Mat object for storing data
        transpose(image,transposed_image);      //transposing given input image
    }
    computeESF(selfThresholdResult);
}

void PDSComputeMTF::computeESF(double selfThresholdResult) {
    // Declare variables
    Mat esfImg = image;
    Mat dst;
    Mat kernel;
    Point anchor;
    double delta;
    int ddepth;
    int rows;
    int columns;

    if (esfImg.empty()) {
        printf(" Error opening image\n");
    }

    // Initialize arguments for the filter
    anchor = Point(-1, -1);
    delta = 0;
    ddepth = -1;

    // Update kernel size for a normalized box filter
    kernel = (Mat_<double>(3, 3)<<0.11111111,0.11111111,0.11111111,
                                           0.11111111,0.11111111,0.11111111,
                                           0.11111111,0.11111111,0.11111111);

    // Apply filter
    filter2D(esfImg, dst, ddepth, kernel, anchor, delta, BORDER_DEFAULT);
    Mat smoothedImage;

    //copy dst to smoothedImage
    dst.copyTo(smoothedImage);

    //find rows and columns of the image
    rows = smoothedImage.rows;
    columns = smoothedImage.cols;

    //clone the image to avoid modifying the original image
    //set size of the image to be the same as the original image
    Mat diffSmoothImage = Mat(rows, columns, CV_64F);
    //set empty values to the image
    diffSmoothImage.setTo(0);

    //convert to double
    smoothedImage.convertTo(smoothedImage, CV_64F);
    esfImg.convertTo(esfImg, CV_64F);

    int tempAppEdge;
    int stripCroppedCounter = 0;
    int absDiffMax = 0;
    int numRowArrPos;
    int numColArrPos;
    int numberBins;
    double difference;
    double boundEdgeLeftExpand;
    double boundEdgeRightExpand;
    double binPad = 0.0001;
    double pixelSubdiv = 0.1;
    double topEdge = 0;
    double bottomEdge = 0;
    double sum = 0;
    Mat temp= Mat(1, 1, CV_64F);
    Mat appEdge= Mat(rows, 1, CV_64F);
    int boundEdgeLeft;
    int boundEdgeRight;
    Mat stripCropped = Mat(rows, 5, CV_64F);
    Mat appEdgeVectorX = Mat(rows, 5, CV_64F);
    Mat appEdgeVectorY = Mat(rows, 5, CV_64F);
    vector<double> tempAppEdgeVectorX = {};
    vector<double> tempAppEdgeVectorY = {};
    vector<double> edgePosition(rows);
    vector< std::vector<double> > arrayValuesNearEdge(rows, std::vector<double>(13));
    vector< std::vector<double> > arrayPositions(rows, std::vector<double>(13));
    vector< std::vector<double> > arrayPositionsByEdge(rows, std::vector<double>(13));
    vector<double> binEdges = {};
    vector<double> binPositions = {};
    vector<double>esf;
    vector<double>xEsf;
    vector<double> esfSmooth;

    for (int i = 0; i < rows; i++) {
        for (int j = 1; j < columns; j++) {
            difference = smoothedImage.at<double>(i,j) - smoothedImage.at<double>(i,j-1);
            //abs value of the difference
            difference = abs(difference);
            diffSmoothImage.at<double>(i,j-1) = difference;
            if (difference > absDiffMax) {
                absDiffMax = difference;
                temp.at<double>(0,0) = diffSmoothImage.at<double>(i,j-1);
                tempAppEdge = j-1;
            }
        }
        if(absDiffMax <= 1){
            cout<<"No edge found"<<endl;
        }
        appEdge.at<double>(i,0) = temp.at<double>(0,0);
        absDiffMax=0;
        boundEdgeLeft= tempAppEdge - 2;
        boundEdgeRight= tempAppEdge + 3;

        //filled appEdgeVectorY and appEdgeVectorX values with Y->[1,2,3,4,5] and X->stripCropped values with the values of the image
        for(int k= boundEdgeLeft ; k < boundEdgeRight ; k++){
           stripCropped.at<double>(i,stripCroppedCounter) = esfImg.at<double>(i,k);
           appEdgeVectorY.at<double>(i,stripCroppedCounter) = stripCroppedCounter+1;
           appEdgeVectorX.at<double>(i,stripCroppedCounter) = stripCropped.at<double>(i,stripCroppedCounter);
           stripCroppedCounter++;
        }
        stripCroppedCounter = 0;

        for (int j = 0; j <appEdgeVectorX.cols; ++j) {
                tempAppEdgeVectorX.push_back(appEdgeVectorX.at<double>(i,j));
                tempAppEdgeVectorY.push_back(appEdgeVectorY.at<double>(i,j));
        }
        tk::spline s(tempAppEdgeVectorX,tempAppEdgeVectorY, tk::spline::cspline);
        edgePosition.at(i) = (s(selfThresholdResult)+boundEdgeLeft-1);
        tempAppEdgeVectorX.clear();
        tempAppEdgeVectorY.clear();

        boundEdgeLeftExpand = tempAppEdge - 6;
        boundEdgeRightExpand = tempAppEdge + 7;

        for (int k = boundEdgeLeftExpand; k < boundEdgeRightExpand; k++) {
            arrayValuesNearEdge[i][k-boundEdgeLeftExpand] = esfImg.at<double>(i,k);
            arrayPositions[i][k-boundEdgeLeftExpand] = k;
        }
    }
    Mat transposedImage;
    Mat npOnes = npOnes.ones(13, 1, CV_64F);
    Mat tempEdgePosition = Mat(13, rows, CV_64F);
    npOnes.setTo(1);
    for (int i = 0; i < 13; i++) {
        for (int k = 0; k < rows; k++) {
            tempEdgePosition.at<double>(i, k) = edgePosition.at(k);
        }
    }
    //transposing given input image
    transpose(tempEdgePosition,transposedImage);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < 13; j++) {
            arrayPositionsByEdge[i][j] = arrayPositions[i][j] - transposedImage.at<double>(i,j);
        }
    }
    numRowArrPos = arrayPositionsByEdge.size();
    numColArrPos = arrayPositionsByEdge[0].size();

    Mat arrayValuesByEdge = Mat (1, numColArrPos*numRowArrPos, CV_64F);
    Mat positionValuesByEdge = Mat (1, numColArrPos*numRowArrPos, CV_64F);
    for (int i = 0; i < numColArrPos; i++) {
        for (int j = 0; j < numRowArrPos; j++) {
            arrayValuesByEdge.at<double>(0,i*numRowArrPos+j) = arrayValuesNearEdge[j][i];
            positionValuesByEdge.at<double>(0,i*numRowArrPos+j) = arrayPositionsByEdge[j][i];
        }
    }
    minMaxLoc(positionValuesByEdge, &bottomEdge, &topEdge);
    topEdge += binPad + pixelSubdiv;
    bottomEdge -= binPad;

    for (double i = bottomEdge; i < topEdge + 1; i += pixelSubdiv) {
        binEdges.push_back(i);
    }

    numberBins = binEdges.size() - 1;
    for (int i = 0; i < numberBins; ++i) {
        binPositions.push_back(binEdges.at(i) + pixelSubdiv / 2);
    }

    vector<double> whichBin = {};
    vector<bool> flagInMembersBool = {};
    vector<double> flagInMembers = {};
    vector<double> binMean(numberBins);
    vector<double> binMembers = {};
    //digitize function
    for (int i = 0; i < positionValuesByEdge.cols; i++) {
        for (int j = 1; j < binEdges.size(); j++) {
            if (positionValuesByEdge.at<double>(0, i) >= binEdges.at(j-1) && positionValuesByEdge.at<double>(0, i) < binEdges.at(j)) {
                whichBin.push_back(j-1);
            }
        }
    }
    for (int i = 0; i < numberBins; ++i) {
        for (int j = 0; j < whichBin.size(); ++j) {
            if (whichBin.at(j) == i) {
                binMembers.push_back(arrayValuesByEdge.at<double>(0, j));
            }
        }
        if (binMembers.size() > 0) {
            sum = 0;
            for (int j = 0; j < binMembers.size(); ++j) {
                sum += binMembers.at(j);
            }
            binMean.at(i) = sum / binMembers.size();
        }
        else
            binMean.at(i) = nan("");
        binMembers.clear();
    }
    binMean = NanHelper(binMean);
    esf=binMean;
    xEsf =binPositions;
    double minEsf = *min_element(xEsf.begin(), xEsf.end());
    for (int i = 0; i < xEsf.size(); ++i) {
        xEsf.at(i) -= minEsf;
    }
    esfSmooth = sg_smooth(esf, 25, 3);
    selfEsf=esf;
    smoothEsf=esfSmooth;
    //print selfESf
    cout << "Self ESF" << endl;
    for (int i = 0; i < selfEsf.size(); ++i) {
        cout << selfEsf.at(i) << " ";
    }
    cout << endl;
    //print smoothESF
    cout << "Smooth ESF" << endl;
    for (int i = 0; i < smoothEsf.size(); ++i) {
        cout << smoothEsf.at(i) << " ";
    }
    cout << endl;
    computeLSF();
}
void PDSComputeMTF::computeLSF() {
    vector<double> diffEsf;
    vector<double> diffEsfSmooth;
    vector<double> lsf;
    vector<double> lsfSmooth;
    double difference=0;
    for (int i = 1; i < selfEsf.size(); i++) {
        difference= abs(selfEsf.at(i) - selfEsf.at(i-1));
        diffEsf.push_back(difference);
    }
    diffEsf.push_back(0);
    lsf=diffEsf;
    for (int i = 1; i < smoothEsf.size(); i++) {
        difference= abs(smoothEsf.at(i-1) - smoothEsf.at(i));
        diffEsfSmooth.push_back(difference);
    }
    diffEsfSmooth.push_back(0);
    lsfSmooth=diffEsfSmooth;
    selfLsf=lsf;
    smoothLsf=lsfSmooth;
    //print selfLSF
    cout<<endl<<"selfLSF"<<endl;
    for (int i = 0; i < selfLsf.size(); i++) {
        cout << selfLsf.at(i) << " ";
    }
    cout<<endl;
    //print smoothLSF
    cout<<endl<<"smoothLSF"<<endl;
    for (int i = 0; i < smoothLsf.size(); i++) {
        cout << smoothLsf.at(i) << " ";
    }
    cout<<endl;
    computeMTF();
}


void PDSComputeMTF::computeMTF() {
    vector<double> mtf;
    vector<double> mtfSmooth;
    vector<double> mtfFinal;
    vector<double> mtfFinalSmooth;
    vector<double> mtfTemp, mtfTempSmooth;
    vector<double> xMtfFinal;
    int mtfMaxValue;
    int mtfMaxValueSmooth;

    //cv::dft(selfLsf, mtf);
    //cv::dft(smoothLsf, mtfSmooth);
    mtf = CalculateDFT(selfLsf, 2048,0);
    mtfSmooth = CalculateDFT(smoothLsf, 2048,0);

    mtfFinal = FftShift(mtf);
    mtfFinalSmooth = FftShift(mtfSmooth);
    mtfMaxValue = *max_element(mtfFinal.begin() + 1024, mtfFinal.begin() + 1151);
    mtfMaxValueSmooth = *max_element(mtfFinalSmooth.begin() + 1024, mtfFinalSmooth.begin() + 1151);
    for (int i = 1024; i < 1151; ++i) {
        mtfTemp.push_back(mtfFinal.at(i)/mtfMaxValue);
        mtfTempSmooth.push_back(mtfFinalSmooth.at(i)/mtfMaxValueSmooth);
    }
    for (int i = 0; i < 127; ++i) {
        xMtfFinal.push_back(i/127.0);
    }

    mtfFinal = mtfTemp;
    mtfFinalSmooth = mtfTempSmooth;
    //plt.plot(xMtfFinal, mtfFinal, 'y-', xMtfFinal, mtfFinalSmooth)
}

int main()
{
    String filePath = "/home/isil/CLionProjects/Project1/33.png";
    Mat croppedImage = ROISelection(filePath);

    PDSComputeMTF pds;
    pds.Init(croppedImage);

    EventHandler myObj;
    myObj.filePath(filePath);
    myObj.LineSelectCallBack();
    myObj.EventExitManager();

    return 0;
}

