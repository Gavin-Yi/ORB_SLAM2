/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}

// 为双目相机重载的构造函数
Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    // Step 1 帧的ID自增
    mnId=nNextId++;

    // Scale Level Info
    // Step 2 计算图像金字塔的参数
    // 获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    // 获得层与层之间的缩放系数
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    // 计算上面缩放比的对数 Notice：log是以自然对数为底的
    mfLogScaleFactor = log(mfScaleFactor);
    // 获得每层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    // 获得每层图像的缩放因子倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    // 高斯模糊的时候使用的方差
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    // Sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // Step 3 提取特征点
    // ORB extraction
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

    // mvKeys中保存的是左图像中的特征点，这里获得左侧图像中特征点的个数
    N = mvKeys.size();

    //如果左侧图像没有提取到特征点就返回，也就意味着这一帧的图像无法使用
    if(mvKeys.empty())
        return;

    // Step 4 用OpenCV的矫正函数，内参进行去畸变
    // 实际上由于双目输入的图像已经预先经过矫正，所以实际上并没有对特征点进行任何处理操作
    UndistortKeyPoints();

    // Step 5 计算双目间特征点的匹配，只有匹配成功的特征点会计算其深度，深度存放在 mvDepth
    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// 为RGBD相机重载的构造函数
Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, 
    cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

// Monocular单目相机的构造函数
Frame::Frame(
    const cv::Mat &imGray,          // 传入的灰度图像
    const double &timeStamp,        // 时间戳
    ORBextractor* extractor,        // ORB提取器
    ORBVocabulary* voc,             // 词典
    cv::Mat &K,                     // 相机内参
    cv::Mat &distCoef,              // 畸变系数
    const float &bf,                // 双目相机的 (基线)X(fx系数)
    const float &thDepth)           // 区分近远点的阈值
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)
{
    // Frame ID
    // ! Frame ID 是一个静态成员变量，作用域为全局
    // Step 1 帧的ID自增
    mnId=nNextId++;

    // Step 2 计算图像金字塔的参数
    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // Step 3 提取特征点
    // ORB extraction
    ExtractORB(0,imGray);   // 0-左眼，1-右眼

    // 求出特征点的个数
    N = mvKeys.size();
    // 如果没有成功提取特征点，则直接返回
    if(mvKeys.empty())
        return;

    // Step 4 用OpenCV的矫正函数，内参进行去畸变
    // 关键点从 mvKeys 到 mvKeysUn
    UndistortKeyPoints();

    // Set no stereo information
    // 单目相机没有右图像和深度信息
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    // 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    // 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
    // Step 5 计算去畸变后图像的边界，将特征点分配到网格当中，这个过程一般只在第一帧或者相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
        // 计算去畸变图像的边界
        ComputeImageBounds(imGray);

        // 一个图像默认是 48(行) * 64(列) 个网格
        // 表示一个图像像素相当于多少个图像网格列
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
        // 表示一个图像像素相当于多少个图像网格行
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

        // 给类的静态成员赋值
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        // ? 可能是因为要频繁使用且耗时，所以保存结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

/**
 * @brief 将提取到的ORB特征点分配到网格当中
 **/
void Frame::AssignFeaturesToGrid()
{
    // Step 1 给存储特征点的网格数组 Frame::mGrid 预分配空间
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    // 对mGrid数组中的每一个vector元素预分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    // Step 2 遍历每个去畸变之后的特征点,将索引保存到网格数组中
    for(int i=0;i<N;i++)
    {
        // 从类的成员变量中，获取每个去畸变之后的特征点
        const cv::KeyPoint &kp = mvKeysUn[i];

        // 计算去畸变之后特征点的网格坐标，如果在网格内，则把索引存入网格数组
        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

/**
 * @brief 提取ORB特征点，通过仿函数，进入ORBextractor文件
 * 
 * @param[in] flag 0-左眼 1-右眼
 * @param[in] im 输入的灰度图像
 **/
void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

// 根据Tcw计算mRcw mtcw 和 mRwc mOw
void Frame::UpdatePoseMatrices()
{ 
    // mOw      当前相机光心在世界坐标系下的坐标
    // mTcw     世界坐标系到相机坐标系的变换矩阵
    // mRcw     世界坐标系到相机坐标系的旋转矩阵
    // mtcw     世界坐标系到相机坐标系的平移向量
    // mRwc     相机坐标系到世界坐标系的旋转矩阵

    // 从变换矩阵中提取旋转矩阵
    // Notice: rowRange 只提取左边界，不提取右边界
    mRcw = mTcw.rowRange(0,3).colRange(0,3);

    // 旋转求逆和求转置等价
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

/**
 * @brief 找到在以 x,y 为中心，半径为r的矩形内且金字塔层级在[minLevel, maxLevel]的特征点
 * 
 * @param [in] x            特征点x坐标
 * @param [in] y            特征点y坐标
 * @param [in] r            矩形搜索半径
 * @param [in] minLevel     最小金字塔层级
 * @param [in] maxLevel     最大金字塔层级
 * @return                  返回搜索到的候选匹配点ID
 **/
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    // 存储搜索结果的vector
    vector<size_t> vIndices;
    vIndices.reserve(N);

    // Step 1 计算半径为r的矩形左右上下边界所在的 网格 列和行的ID
    // mfGridElementWidthInv = (mfGridElementWidthInv) / (mnMaxX - mnMinX) ，表示一个元素占多少个网格(肯定小于1)
    // (x-mnMinX-r)计算的圆最左侧的点的像素坐标，乘上 mfGridElementWidthInv,表示圆最左侧所在的网格的列ID
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        // 出错，直接返回空的vector
        return vIndices;
    // 计算圆右边界所在网格的列ID
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;
    // 计算圆上边界所在网格的行ID
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;
    // 计算圆下边界所在网格的行ID
    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    // 检查需要搜索的图像金字塔层数是否符合要求
    // ?疑似bug，后面的 maxLevel>=0 肯定True
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    // Step 2 遍历矩形区域内的所有网格，寻找满足条件的候选特征点，并将其index放到输出里
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            // 获取这个网格内的所有特征点在 Frame::mvKeyUn 中的索引
            const vector<size_t> vCell = mGrid[ix][iy];
            // 如果这个网格中没有特征点，那么跳过这个网格继续下一个
            if(vCell.empty())
                continue;

            // 如果这个网格中有特征点，那么遍历这个图像网格中所有的特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                // 根据索引先读取这个特征点
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                // 保证给定的搜索金字塔层级范围合法
                if(bCheckLevels)
                {
                    // cv::KeyPoint::octave 表示的是从金字塔的哪一层抽取的数据
                    // 保证特征点层级是在金字塔层级 minLevel 和 maxLevel 之间
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                // 计算特征点到圆的中心的距离
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                // 如果x方向和y方向的距离都在半径之内，存储它的index作为候选点
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

/**
 * @brief
 * 
 * @param[in] kp            给定的特征点
 * @param[in out] posX      特征点所在网格坐标的横坐标
 * @param[in out] posY      特征点所在网格坐标的纵坐标
 * @return true             如果找到特征点所在的网格坐标，返回true
 * @return false            没有找到，返回false
 **/
bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    // 计算特征点落在了哪个网格
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    // 因为特征点进行了去畸变，而且前面进行了round操作，所以算出的网格坐标，可能会超出实际划分的网格
    // 如果计算的网格坐标超出了可能的坐标，返回false
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief 计算当前帧特征点对应的词袋Bow，主要是mBowVec 和 mFeatVec
 **/
void Frame::ComputeBoW()
{
    // 判断是否以前已经计算过了，计算过了就跳过
    if(mBowVec.empty())
    {
        // 将描述子 mDescriptor 转换为 DBow 要求的输入格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        // 将特征点的描述子转换成词袋向量 mBowVec 以及特征向量 mFeatVec
        mpORBvocabulary->transform( vCurrentDesc,   // 当前的描述子vector
                                    mBowVec,        // 输出，词袋向量，记录的是单词的ID及其对应权重TF-IDF值
                                    mFeatVec,       // 输出，记录node id 及其对应的图像 feature对应的索引
                                    4);             // 4 表示从叶节点向前数的层数
    }
}

/**
 * @brief 用内参对特征点进行去畸变
 **/
void Frame::UndistortKeyPoints()
{
    // 如果畸变参数k1为0，则后面的参数基本都为0，直接返回
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    // N为提取的特征点数量，为了满足OpenCV的函数输入要求，创建一个N*2维的矩阵
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        // 遍历每个特征点，然后将他们的横纵坐标分别保存
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    // reshape函数将mat变为2通道的矩阵，行数保持不变
    // OpenCV的去畸变函数需要mat具有2通道
    mat=mat.reshape(2);
    cv::undistortPoints(
        mat,            // 输入的特征点坐标
        mat,            // 输出的校正后的特征点坐标覆盖原矩阵
        mK,             // 内参矩阵
        mDistCoef,      // 畸变参数
        cv::Mat(),      // 一个空矩阵，原作用是矫正
        mK);            // 新内参矩阵

    // 将矩阵恢复为单通道
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        // 遍历每个特征点，更改他们的坐标值
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        // 将结果存入mvKeysUn向量
        mvKeysUn[i]=kp;
    }
}

/**
 * @brief 计算去畸变图像的边界
 * 
 * @param[in] imLeft    需要计算边界的图像
 **/
void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    // 如果畸变参数不为0
    if(mDistCoef.at<float>(0)!=0.0)
    {
        // 保存校正前的图像四个边界点坐标
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;                     // 左上
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;             // 右上
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;             // 左下
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;     // 右下

        // Undistort corners
        // 和前面校正特征点的操作一样，把四个顶点作为内容去畸变
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        // 取左上和左下的点的横坐标的较小值为MinX
        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        // 取右上和右下的点的横坐标的较大值为MaxX
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        // 取左上和右上的点的纵坐标的较小值为MinY
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        // 取左下和右下的点的纵坐标的较大值为MaxY
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        // 如果畸变参数为0，默认以当前图像大小为边界
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/**
 * @brief 双目相机的立体匹配
 * 
 * 为左图的每一个特征点在右图中找到匹配点
 * 根据基线上描述子距离找到匹配，在进行SAD精确定位
 * 这里所说的SAD是一种双目立体视觉匹配算法，可参考[https://blog.csdn.net/u012507022/article/details/51446891]
 * 最后对所有的SAD的值进行排序，剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配
 * 这里所谓的亚像素精度，就是使用这个拟合得到一个小于一单位像素的修正量，这样可以取得更好的估计结果，计算出来的点的深度也就
 * 越准确，匹配成功后更新 mvuRight(ur) 和 mvDepth(Z)
 **/
void Frame::ComputeStereoMatches()
{
    /* 两帧图像稀疏立体匹配(即：ORB特征点匹配，非逐像素的密集匹配，但依然满足行对齐)
        输入：两帧立体矫正后的图像 img_left 和 img_right 对应的ORB特征点集
        过程：
            1. 行特征点统计，统计 img_right 每一行上的ORB特征点集，便于使用立体匹配思路(行搜索/极线搜索)进行同名搜索，避免逐像素的判断
            2. 粗匹配，根据步骤1的结果，对 img_left 第 i 行的ORB特征点pi，在 img_right 的第i行上的ORB特征点集中搜索相似ORB特征点，得到qi
            3. 精确匹配，以点qi为中心，半径为r的范围内，进行块匹配(归一化SAD)，进一步优化匹配结果
            4. 亚像素精度优化，步骤3得到的视差为uchar/int类型精度，并不一定是真实视差，通过亚像素差值(抛物线插值)获取float精度的真实视差
            5. 最优视差值/深度选择，通过胜者为王算法(WTA)获取最佳匹配点
            6. 删除离群点(Outliers),块匹配相似度阈值判断，归一化SAD最小，并不代表就一定是正确匹配，比如光照变化、弱纹理等会造成误匹配
        输出：稀疏特征点视差图/深度图(亚像素精度) mvDepth 匹配结果 mvuRight
     */

    // 为存储结果预先分配内存，数据类型为Float型
    // mvuRight存储右图匹配点索引
    // mvDepth存储特征点的深度信息
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    // ORB特征相似度阈值
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    // 图像金字塔顶层（0层）的图像高
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    // 二维vector存储每一行的ORB特征点的列坐标
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    // 右图特征点数量，N表示数量，r表示右图，const表示不能被修改
    const int Nr = mvKeysRight.size();

    // Step 1 行特征点统计，考虑到尺度金字塔特征，一个特征点可能存在于多行，而非唯一的一行
    for(int iR=0; iR<Nr; iR++)
    {
        // 获取特征点ir的y坐标，即行号
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;

        // 计算特征点ir在行方向上，可能的偏移范围r，即可能的行号为[kpY + r, kpY - r]
        // 2 表示在全尺寸(scale - 1)的情况下，假设有2个像素的偏移，随着尺度变化，r也跟着变化
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        // 将特征点ir保存在可能的行号中
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Step 2  粗匹配 + 精匹配
    // 对于立体矫正后的两张图，在列方向(x)存在最大视差maxd和最小视察mind
    // 也就是左图中的任何一点p，在右图上的匹配点的范围应该是[p - maxd, p - mind],而不需要遍历每一行的所有元素
    // maxd = baseline * length_focal / minz
    // mind = baseline * length_focal / maxz

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // 保存SAD块匹配相似度和左图特征点索引
    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    // 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        // 获取左图特征点il在右图对应行中可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        // 计算理论上的最佳搜索范围
        const float minU = uL-maxD;
        const float maxU = uL-minD;

        // 最大搜索范围小于0，说明无匹配点
        if(maxU<0)
            continue;

        // 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Step 2.1 粗匹配，左图特征点il与右图中的可能的匹配点进行逐个比较，得到最相似匹配点的相似度和索引
        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            // 左图特征点il与待匹配点ic的空间尺度差超过2，放弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            // 使用列坐标(x)进行匹配，和stereomatch一样
            const float &uR = kpR.pt.x;

            // 超出理论搜索范围[minU maxU]，可能是误匹配，放弃
            if(uR>=minU && uR<=maxU)
            {
                // 计算匹配点il 和待匹配点ic的相似度dist
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                // 统计最小相似度及其对应的列坐标(x)
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // 如果上述匹配过程中的最佳描述子距离小于给定的阈值，就进行精确匹配
        // Step 3 精确匹配
        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            // 计算右图特征点x坐标和对应的金字塔尺度
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];

            // 尺度缩放之后的左右图特征点的坐标
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            // 滑动窗口搜索，类似模板卷积或滤波， w是SAD相似的窗口半径
            const int w = 5;

            // 提取左图中，以特征点(scaledul,scaledvl)为中心，半径为w的图像块patch
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);

            // 图像块减去了中心的像素，降低亮度变化对相似度计算的影响
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            // 初始化最佳相似度
            int bestDist = INT_MAX;

            // 通过滑动窗口搜索优化，得到的列坐标偏移量
            int bestincR = 0;

            // 滑动窗口的滑动范围为 (-L, L)
            const int L = 5;

            // 初始化存储图像块相似度
            vector<float> vDists;
            vDists.resize(2*L+1);

            // 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
            // 列方向七点 iniU = r0 + 最大窗口滑动范围 - 图像块尺寸
            // 列方向重点 eniU = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;

            // 判断搜索是否越界
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            // 在搜索范围内从左到右滑动，并计算图像块相似度
            for(int incR=-L; incR<=+L; incR++)
            {
                // 提取右图时，以特征点[scaleduL, scaledvL]为中心，半径为w的图像块patch
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);

                // 图像块减去了中心的像素，降低亮度变化对相似度计算的影响
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                // SAD计算
                float dist = cv::norm(IL,IR,cv::NORM_L1);
                // 统计最小SAD和偏移量
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            // 搜索窗口越界判断
            if(bestincR==-L || bestincR==L)
                continue;

            // Step 4 亚像素插值，使用最佳匹配点及其左右相邻点构成抛物线
            // 使用3点拟合抛物线的方式，用极小值代替之前计算的最优视差值
            // 计算公式参考论文 <On Building an Accurate Stereo Matching System on Graphics Hardware> 公式7
            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));
            // 亚像素精度的偏移量应该是在[-1, 1]之间，否则就是误匹配
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            // 根据亚像素精度偏移量 delta 调整最佳匹配索引
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);
            // 检查加上亚像素偏移之后的视差是否还在规定的范围之内
            if(disparity>=minD && disparity<maxD)
            {
                // 如果存在负视差，则约束为0.01
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                // 根据视差值计算深度信息
                // 保存相似点的列坐标信息
                // 保存归一化SAD最小相似度
                // Step 5 最优视差值/深度选择
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    // Step 6 删除离群点(Outliers)
    // 块匹配度相似度阈值判断，归一化SAD最小，并不代表一定就是匹配的，比如光照变化，弱纹理，无纹理等同样会造成误匹配
    // 误匹配判断条件 norm_sad > 1.5 * 1.4 * median
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;     // 中值
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

// 把当前帧中的特征点反投影为3D点
cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)     // 应该具有正的深度值，单目初始化为-1
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
