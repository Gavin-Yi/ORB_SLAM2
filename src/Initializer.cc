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

#include "Initializer.h"

#include "Thirdparty/DBoW2/DUtils/Random.h"

#include "Optimizer.h"
#include "ORBmatcher.h"

#include<thread>

namespace ORB_SLAM2
{

/**
 * @brief 初始化器的构造函数
 * @param ReferenceFrame 参考帧
 * @param sigma 标准差?
 * @param iterations 最大迭代次数
 **/
Initializer::Initializer(const Frame &ReferenceFrame, float sigma, int iterations)
{
    // 相机的内参矩阵
    mK = ReferenceFrame.mK.clone();

    // 保存参考帧的所有关键点
    mvKeys1 = ReferenceFrame.mvKeysUn;

    mSigma = sigma;
    mSigma2 = sigma*sigma;

    // 最大迭代次数
    mMaxIterations = iterations;
}

/**
 * @brief 计算基础矩阵和单应矩阵,选取最佳的来会付出最开始两帧之间的相对姿态，并进行三角化得到初始地图点
 * 
 * @param [in] CurrentFrame         当前帧，也就是SLAM意义上的第二帧
 * @param [in] vMatches12           当前帧(2)和参考帧(1)图像中特征点的匹配关系
 *                                  vMatches12[i]解释：i表示帧1中关键点的索引值，vMatches12[i]为帧2中关键点的索引值
 *                                  如果没有匹配关系，则vMatches12[i]的值为-1
 * @param [out] R21                 相机从参考帧到当前帧的旋转
 * @param [out] t21                 相机从参考帧到当前帧的平移
 * @param [out] vP3D                三角化测量之后的三维地图点
 * @param [out] vbTriangulated      标记三角化点是否有效
 * @return true     该帧可以成功初始化
 * @return false    该帧不能成功初始化
 **/
bool Initializer::Initialize(
    const Frame &CurrentFrame, 
    const vector<int> &vMatches12, 
    cv::Mat &R21, 
    cv::Mat &t21,
    vector<cv::Point3f> &vP3D, 
    vector<bool> &vbTriangulated)
{
    // Fill structures with current keypoints and matches with reference frame
    // Reference Frame: 1, Current Frame: 2

    // 获取当前帧去畸变后的特征点
    mvKeys2 = CurrentFrame.mvKeysUn;

    // mvMatches12 记录的是帧2在帧1中匹配的索引，预分配空间
    mvMatches12.clear();
    mvMatches12.reserve(mvKeys2.size());

    // 记录帧1中每个点是否有匹配的特征点
    // 这个成员变量后面没有用到，后面只关心匹配上的特征点
    mvbMatched1.resize(mvKeys1.size());

    // Step 1 重新记录特征点对的匹配关系，存储在 mvMatches12 中，是否有匹配存储在 mvbMatched1
    for(size_t i=0, iend=vMatches12.size();i<iend; i++)
    {
        // 没有匹配关系的话，值为-1
        if(vMatches12[i]>=0)
        {
            // i表示帧1中关键点的索引值，vMatches12[i]为帧2关键点的索引值
            mvMatches12.push_back(make_pair(i,vMatches12[i]));
            mvbMatched1[i]=true;
        }
        else
            // 标记帧1中改点没有对应的匹配关系
            mvbMatched1[i]=false;
    }

    // 有匹配的特征点的对数
    const int N = mvMatches12.size();

    // Indices for minimum set selection
    // 新建一个vAllIndices容器存储特征点索引，并预分配空间
    vector<size_t> vAllIndices;
    vAllIndices.reserve(N);

    // 在RANSAC的某次迭代中，还可以被抽取作为数据样本的特征点对的索引，所以叫做可用索引
    vector<size_t> vAvailableIndices;
    // 初始化所有特征点对的索引值
    for(int i=0; i<N; i++)
    {
        vAllIndices.push_back(i);
    }

    // Generate sets of 8 points for each RANSAC iteration
    // Step 2 在所有匹配的特征点对中，随机选择8对匹配特征点，用于估计H矩阵和F矩阵
    // 选择 mMaxIterations 组(默认200)
    // mvSets 用于保存每次迭代时使用的向量
    mvSets = vector< vector<size_t> >(mMaxIterations,           // 最大的RANSAC迭代次数
                                        vector<size_t>(8,0));   // 初始值为0的包含8个数的向量

    // 用于进行随机数据样本采样,设置随机数种子
    DUtils::Random::SeedRandOnce(0);

    // 开始每一次的迭代
    for(int it=0; it<mMaxIterations; it++)
    {
        // 迭代开始时，所有点都是可用的
        vAvailableIndices = vAllIndices;

        // Select a minimum set
        // 使用最小的数据样本集，也就是八点法求
        for(size_t j=0; j<8; j++)
        {
            // 随机产生一对点的id，范围从0到N-1
            int randi = DUtils::Random::RandomInt(0,vAvailableIndices.size()-1);
            // idx表示哪一个索引对应的特征点被选中
            int idx = vAvailableIndices[randi];

            // 将本次迭代的第k个特征点存入容器
            mvSets[it][j] = idx;

            // 由于这个点在本次迭代中已被使用，为了避免被重复选中，用vAvailableIndices中的最后一个元素
            // 覆盖掉这个点目前所在的位置，并把最后一个元素删除,避免最后一个点出现两次
            vAvailableIndices[randi] = vAvailableIndices.back();
            vAvailableIndices.pop_back();
        }   // 依次提取出8个特征点对
    }   // 迭代200次，选取各自迭代时需要用到的最小数据集



    // Launch threads to compute in parallel a fundamental matrix and a homography
    // Step 3 计算 fundamental 矩阵和 homography 矩阵，为了加速分别开了线程计算

    vector<bool> vbMatchesInliersH, vbMatchesInliersF;
    // 计算出来的单应矩阵和基础矩阵的RANSAC得分，使用的是重投影误差
    float SH, SF;
    // 经过RANSAC后计算出来的单应矩阵和基础矩阵
    cv::Mat H, F;

    // 构造线程来计算H矩阵及其得分
    thread threadH(
        &Initializer::FindHomography,   // 该线程的主函数
        this,                           // 由于主函数为类的成员函数，所以第一个参数应该是当前对象的this指针
        ref(vbMatchesInliersH),         // 输出，特征点的Inlier标记
        ref(SH),                        // 输出，计算的单应矩阵的RANSAC评分
        ref(H));                        // 输出，计算的单应矩阵结果
    // 构造线程计算F矩阵及其得分
    thread threadF(
        &Initializer::FindFundamental,  // 该线程的主函数
        this,                           // 由于主函数为类的成员函数，所以第一个参数应该是当前对象的this指针
        ref(vbMatchesInliersF),         // 输出，特征点的Inlier标记
        ref(SF),                        // 输出，计算的基础矩阵的RANSAC评分
        ref(F));                        // 输出，计算的基础矩阵结果

    // Wait until both threads have finished
    // 等待两个计算线程结束
    threadH.join();
    threadF.join();

    // Compute ratio of scores
    // Step 4 计算得分比例来判断选取哪个模型来求位姿R，t
    float RH = SH/(SH+SF);

    // Try to reconstruct from homography or fundamental depending on the ratio (0.40-0.45)
    // Notice 这里更倾向于用H矩阵来恢复位姿
    if(RH>0.40)
        return ReconstructH(vbMatchesInliersH,      // 匹配成功的特征点对的Inliers标记
                            H,                      // 单应矩阵
                            mK,                     // 相机内参矩阵
                            R21,                    // 输出，旋转矩阵
                            t21,                    // 输出平移向量
                            vP3D,                   // 经过三角测量，存储空间点的向量
                            vbTriangulated,         // 特征点对是否被三角化的标记
                            1.0,                    // 三角测量中，任务测量有效时应某事的最小视差角(视差过小，会引起非常大的误差)，单位是角度
                            50);                    // 为了进行运动恢复，所需要的最少的三角化测量成功的点的个数
    else //if(pF_HF>0.6)
        return ReconstructF(vbMatchesInliersF,F,mK,R21,t21,vP3D,vbTriangulated,1.0,50);

    // 如果程序执行到了这里，说明程序跑飞了，无法获得相邻的相机变换姿态
    return false;
}

/**
 * @brief 计算单应矩阵，假设场景为平面情况下的计算
 * Step 1 将当前帧和参考帧中的特征点坐标进行归一化
 * Step 2 选择8个归一化之后的点进行迭代
 * Step 3 八点法计算单应矩阵
 * Step 4 利用重投影误差为当前RANSAC的结果评分
 * Step 5 更新具有最优评分的单应矩阵结果，并且保存所对应的特征点点对的内点标记
 * 
 * @param vbMatchesInliers  标记结果是否为外点
 * @param score             计算单应矩阵的得分
 * @param H21               单应矩阵结果
 **/
void Initializer::FindHomography(vector<bool> &vbMatchesInliers, float &score, cv::Mat &H21)
{
    // Number of putative matches
    // 匹配的特征点的总数
    const int N = mvMatches12.size();

    // Normalize coordinates
    // Step 1 将当前帧和参考帧中的特征点坐标进行归一化
    // 具体来说，就是mvKeys1和mvKeys2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1，T2
    // 这里所谓的一阶矩其实就是随机变量到取值的中心的绝对值的平均值
    // 归一化矩阵就是把上述归一化的操作用矩阵来表示，这样特征点坐标乘归一化矩阵就可以得到归一化后的坐标

    // 归一化后的参考帧1和当前帧2中的特征点坐标
    vector<cv::Point2f> vPn1, vPn2;
    // 归一化矩阵
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);

    // !这里求的逆在后面的代码中有用到，辅助进行原始尺度的恢复
    cv::Mat T2inv = T2.inv();

    // Best Results variables
    // 记录最佳评分
    score = 0.0;
    // 取历史最佳评分时，特征点对的inliers标记
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    // 某次迭代中，参考帧的特征点坐标
    vector<cv::Point2f> vPn1i(8);
    // 某次迭代中，当前帧的特征点坐标
    vector<cv::Point2f> vPn2i(8);
    // 计算出来的单应矩阵、及其逆矩阵
    cv::Mat H21i, H12i;

    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    // 开始进行每次的RANSAC迭代
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        // Step 2 选择8个归一化后的点开始迭代
        for(size_t j=0; j<8; j++)
        {
            // 从mvSets中获取当前次迭代某个特征点对的索引信息
            int idx = mvSets[it][j];

            // 根据这个特征点对的索引信息分别找到两个特征点在各自图像中的索引。然后读取归一化后的坐标
            vPn1i[j] = vPn1[mvMatches12[idx].first];        // first存储参考帧1中的索引
            vPn2i[j] = vPn2[mvMatches12[idx].second];       // second存储当前帧2中的索引
        }

        // Step 3 八点法计算单应矩阵
        // 在<计算机视觉中的多视图几何>这本书中P193提到,八点法成功的关键在于对输入数据上归一化，后面又恢复这个尺度
        cv::Mat Hn = ComputeH21(vPn1i,vPn2i);

        // 单应矩阵原理 x2' = H21 * x1', x2' 和 x1' 是归一化后的点坐标
        // 归一化 x2' = T2 * x2, x1' = T1 * x1
        H21i = T2inv*Hn*T1;
        H12i = H21i.inv();

        // Step 4 利用重投影误差计算当次RANSAC的结果评分
        currentScore = CheckHomography(
            H21i,                   // 输入，单应矩阵的计算结果  
            H12i,                   // 同上，对应得是没有进行归一化的坐标
            vbCurrentInliers,       // 特征点对的Inliers标记
            mSigma);                // 测量误差，在Initializer类对象构造的时候，由外部给定

        // Step 5 更新具有最优评分的单应矩阵计算结果，并且保存所对应的特征点对的Inliers标记
        if(currentScore>score)
        {
            // 如果当前的结果得分更高，那么就更新最优计算结果
            H21 = H21i.clone();
            // 保存匹配好的特征点对的Inliers标记
            vbMatchesInliers = vbCurrentInliers;
            // 更新历史最优评分
            score = currentScore;
        }
    }
}

/**
 * @brief 计算基础矩阵，有平移和旋转情况下的计算
 * Step 1 将当前帧和参考帧中的特征点坐标进行归一化
 * Step 2 选择8个归一化之后的点进行迭代
 * Step 3 八点法计算基础矩阵
 * Step 4 利用重投影误差为当前RANSAC的结果评分
 * Step 5 更新具有最优评分的基础矩阵结果，并且保存所对应的特征点点对的内点标记
 * 
 * @param vbMatchesInliers  标记结果是否为外点
 * @param score             计算单应矩阵的得分
 * @param H21               单应矩阵结果
 **/
void Initializer::FindFundamental(
    vector<bool> &vbMatchesInliers, 
    float &score, 
    cv::Mat &F21)
{
    // Number of putative matches
    // 匹配的特征点的总数
    const int N = vbMatchesInliers.size();
    
    // Normalize coordinates
    // Step 1 将当前帧和参考帧中的特征点坐标进行归一化
    // 具体来说，就是mvKeys1和mvKeys2归一化到均值为0，一阶绝对矩为1，归一化矩阵分别为T1，T2
    // 这里所谓的一阶矩其实就是随机变量到取值的中心的绝对值的平均值
    // 归一化矩阵就是把上述归一化的操作用矩阵来表示，这样特征点坐标乘归一化矩阵就可以得到归一化后的坐标

    // 归一化后的参考帧1和当前帧2中的特征点坐标
    vector<cv::Point2f> vPn1, vPn2;
    // 归一化矩阵
    cv::Mat T1, T2;
    Normalize(mvKeys1,vPn1, T1);
    Normalize(mvKeys2,vPn2, T2);

    // !这里求的是归一化矩阵的转置
    cv::Mat T2t = T2.t();

    // Best Results variables
    // 记录最佳评分
    score = 0.0;
    // 取历史最佳评分时，特征点对的inliers标记
    vbMatchesInliers = vector<bool>(N,false);

    // Iteration variables
    // 某次迭代中，参考帧的特征点坐标
    vector<cv::Point2f> vPn1i(8);
    // 某次迭代中，当前帧的特征点坐标
    vector<cv::Point2f> vPn2i(8);
    // 计算出来的基础矩阵
    cv::Mat F21i;

    vector<bool> vbCurrentInliers(N,false);
    float currentScore;

    // Perform all RANSAC iterations and save the solution with highest score
    // 开始进行每次的RANSAC迭代
    for(int it=0; it<mMaxIterations; it++)
    {
        // Select a minimum set
        // Step 2 选择8个归一化后的点开始迭代
        for(int j=0; j<8; j++)
        {
            // 从mvSets中获取当前次迭代某个特征点对的索引信息
            int idx = mvSets[it][j];

            // 根据这个特征点对的索引信息分别找到两个特征点在各自图像中的索引。然后读取归一化后的坐标
            vPn1i[j] = vPn1[mvMatches12[idx].first];
            vPn2i[j] = vPn2[mvMatches12[idx].second];
        }

        // Step 3 八点法计算基础矩阵
        cv::Mat Fn = ComputeF21(vPn1i,vPn2i);

        // 基础矩阵约束为: p2^T * F21 * p1 = 0, p1 和 p2 是齐次化的特征点坐标
        // 特征点归一化 vPn1i = T1 * vPn1, vPn2i = T2 * vPn2
        // vPn2^T * T2^T * F21 * T1 * vPn1 = 0
        // 中间的部分 T2^T * F21 * T1 就是 F21i
        F21i = T2t*Fn*T1;

        // Step 4 利用重投影误差计算当次RANSAC的结果评分
        currentScore = CheckFundamental(F21i, vbCurrentInliers, mSigma);

        // Step 5 更新具有最优评分的基础矩阵结果，并且保存所对应的特征点点对的内点标记
        if(currentScore>score)
        {
            F21 = F21i.clone();
            vbMatchesInliers = vbCurrentInliers;
            score = currentScore;
        }
    }
}

/**
 * @brief 用DLT方法计算单应矩阵H
 * 这里至少用4对点就能够求出来，不过这里为了同一还是使用了8对点来求最小二乘解
 * 
 * @param[in] vP1       参考帧中归一化后的特征点坐标
 * @param[in] vP2       当前帧中归一化后的特征点坐标
 * @return cv::Mat      计算的单应矩阵
 **/
cv::Mat Initializer::ComputeH21(const vector<cv::Point2f> &vP1, const vector<cv::Point2f> &vP2)
{
    // |x'|     | h1 h2 h3 | |x|
    // |y'| = a | h4 h5 h6 | |y|    简写： x' = a H x, a是一个尺度因子
    // |1 |     | h7 h8 h9 | |1|
    // 使用DLT(Direct Linear Transform)求解该方程
    // x' = a H x
    // ---> (x')^(H x) = 0
    // ---> A h = 0
    // A = |  0  0  0 -x -y -1 xy' yy' y'|     h = |h1 h2 h3 h4 h5 h6 h7 h8 h9|
    //     | -x -y -1  0  0  0 xx' yx' x'|
    // 通过SVD求解 A h = 0，A^T*A最小特征值对应的特征向量即为解
    // 其实也就是右奇异值矩阵的最后一列

    // 获取参与计算的特征点的数目
    const int N = vP1.size();

    // 构造用于计算的矩阵A
    cv::Mat A(
        2*N,        // 行，每个点对的数据对应两行
        9,          // 列
        CV_32F);    // 数据类型

    for(int i=0; i<N; i++)
    {
        // 获取特征点对的X，Y坐标
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        // 生成这对点的第一行
        A.at<float>(2*i,0) = 0.0;
        A.at<float>(2*i,1) = 0.0;
        A.at<float>(2*i,2) = 0.0;
        A.at<float>(2*i,3) = -u1;
        A.at<float>(2*i,4) = -v1;
        A.at<float>(2*i,5) = -1;
        A.at<float>(2*i,6) = v2*u1;
        A.at<float>(2*i,7) = v2*v1;
        A.at<float>(2*i,8) = v2;

        // 生成这对点的第二行
        A.at<float>(2*i+1,0) = u1;
        A.at<float>(2*i+1,1) = v1;
        A.at<float>(2*i+1,2) = 1;
        A.at<float>(2*i+1,3) = 0.0;
        A.at<float>(2*i+1,4) = 0.0;
        A.at<float>(2*i+1,5) = 0.0;
        A.at<float>(2*i+1,6) = -u2*u1;
        A.at<float>(2*i+1,7) = -u2*v1;
        A.at<float>(2*i+1,8) = -u2;

    }

    // 定义输出变量，u是左边的正交阵，w是奇异值矩阵，vt中的t表示是右正交矩阵的转置
    cv::Mat u,w,vt;

    cv::SVDecomp(A,         // 输入，待进行奇异值分解的矩阵
                w,          // 输出，奇异值矩阵
                u,          // 输出，左正交阵
                vt,         // 输出，右正交阵
                cv::SVD::MODIFY_A | cv::SVD::FULL_UV);  // MODIFY_A是为了加速计算，FULL_UV使输出的U，V单位正交矩阵

    // 返回最小奇异值所对应的右奇异向量
    // 也就是右正交阵的最后一行，第9行，作为最优解。
    // Notice: 奇异值矩阵非负，默认按照对角线从大到小的顺序排列，越靠后，奇异值越小
    // (A - t * I) * h = 0 ---> 特征值t越小，等式越接近于 ---> A * h = 0
    // reshape 将向量还原为一个 3*3 的矩阵
    return vt.row(8).reshape(0, 3);
}

/**
 * @brief 用DLT方法计算基础矩阵F
 * 使用了8对点来求最小二乘解
 * 
 * @param[in] vP1       参考帧中归一化后的特征点坐标
 * @param[in] vP2       当前帧中归一化后的特征点坐标
 * @return cv::Mat      计算的单应矩阵
 **/
cv::Mat Initializer::ComputeF21(const vector<cv::Point2f> &vP1,const vector<cv::Point2f> &vP2)
{
    // 获取匹配特征点对的个数
    const int N = vP1.size();

    // 初始化A矩阵，N*9维
    cv::Mat A(N,9,CV_32F);

    // 一对点生成A矩阵的一行
    for(int i=0; i<N; i++)
    {
        const float u1 = vP1[i].x;
        const float v1 = vP1[i].y;
        const float u2 = vP2[i].x;
        const float v2 = vP2[i].y;

        A.at<float>(i,0) = u2*u1;
        A.at<float>(i,1) = u2*v1;
        A.at<float>(i,2) = u2;
        A.at<float>(i,3) = v2*u1;
        A.at<float>(i,4) = v2*v1;
        A.at<float>(i,5) = v2;
        A.at<float>(i,6) = u1;
        A.at<float>(i,7) = v1;
        A.at<float>(i,8) = 1;
    }

    // 定义输出变量，u是左边的正交阵，w是奇异值矩阵，vt中的t表示是右正交矩阵的转置
    cv::Mat u,w,vt;

    // SVD分解，同H矩阵分解
    cv::SVDecomp(A,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 转换成基础矩阵3*3的形式
    cv::Mat Fpre = vt.row(8).reshape(0, 3);

    // 基础矩阵的秩为2，根据<视觉SLAM十四讲>，本质矩阵的奇异值应该为[u u 0]的情况
    // 通过第二次奇异值分解，来强制使其秩为2
    cv::SVDecomp(Fpre,w,u,vt,cv::SVD::MODIFY_A | cv::SVD::FULL_UV);

    // 强制将第三个奇异值设置为0
    w.at<float>(2)=0;

    // 重新组合好满足秩约束的基础矩阵，作为最终计算结果返回
    return  u*cv::Mat::diag(w)*vt;
}

/**
 * @brief 对给定的Homography矩阵打分，需要用到卡方检验的知识
 * 
 * @param[in] H21                       从参考帧到当前帧的单应矩阵
 * @param[in] H12                       从当前帧到参考帧的单应矩阵
 * @param[in] vbMatchesInliers          匹配好的特征点对的Inliers标记
 * @param[in] sigma                     标准差，默认为1
 * @return                              返回得分
 **/
float Initializer::CheckHomography(
    const cv::Mat &H21, 
    const cv::Mat &H12, 
    vector<bool> &vbMatchesInliers, 
    float sigma)
{   
    // Notice：在已知n维观测数据服从N(0, sigma)的高斯分布时
    // 其误差加权最小二乘结果为 sum_error = SUM(e(i)^T * Q^(-1) * e(i))
    // 其中: e(i) = [e_x, e_y,...]^T, Q为观测数据的协方差矩阵，即 sigma *sigma 组成的协方差矩阵
    // 误差加权最小二次结果越小，说明观测数据精度越高

    // 通过H矩阵，进行参考帧和当前帧之间的双向投影，计算起加权最小二乘投影误差

    // 算法流程:
    // input: 单应性矩阵 H21, H12 ;匹配点集 mvKeys1, mvKeys2; 记录匹配的 mvMatches12
    //    do:
    //          for p1(i), p2(i) in mvKeys:
    //              error_i1 = || p2(i) - H21 * p1(i) ||2
    //              error_i2 = || p1(i) - H12 * p2(i) ||2
    //
    //              w1 = 1 / sigma / sigma
    //              w2 = 1 / sigma / sigma
    //
    //              if error_i1 < th
    //                  score += th - error_i1 * w1
    //              if error_i2 < th
    //                  score += th - error_i2 * w2
    //
    //              if error_i1 > th or error_i2 > th
    //                  p1(i),p2(i) are outer points
    //                  vbMatchesInliers(i) = false
    //              else
    //                  p1(i),p2(i) are inner points
    //                  vbMatchesInliers(i) = true
    //              end
    //          end
    //   output: score, inliers

    // 特征点匹配的个数
    const int N = mvMatches12.size();

    // 获取从参考帧到当前帧的单应矩阵的各个元素
    const float h11 = H21.at<float>(0,0);
    const float h12 = H21.at<float>(0,1);
    const float h13 = H21.at<float>(0,2);
    const float h21 = H21.at<float>(1,0);
    const float h22 = H21.at<float>(1,1);
    const float h23 = H21.at<float>(1,2);
    const float h31 = H21.at<float>(2,0);
    const float h32 = H21.at<float>(2,1);
    const float h33 = H21.at<float>(2,2);

    // 获取从当前帧到参考帧的单应矩阵的各个元素
    const float h11inv = H12.at<float>(0,0);
    const float h12inv = H12.at<float>(0,1);
    const float h13inv = H12.at<float>(0,2);
    const float h21inv = H12.at<float>(1,0);
    const float h22inv = H12.at<float>(1,1);
    const float h23inv = H12.at<float>(1,2);
    const float h31inv = H12.at<float>(2,0);
    const float h32inv = H12.at<float>(2,1);
    const float h33inv = H12.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 基于卡方检验计算出的阈值(假设测量有一个pixel的偏差)
    // 自由度为2的卡方分布，显著性水平为0.05，对应的阈值
    const float th = 5.991;

    // 信息矩阵，方差平方的倒数
    const float invSigmaSquare = 1.0/(sigma*sigma);

    for(int i=0; i<N; i++)
    {
        // 一开始都默认为Inliers
        bool bIn = true;

        // 提取参考帧和当前帧之间匹配的特征点
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in first image
        // x2in1 = H12*x2
        // 计算 img2 到 img1 的重投影误差
        // |u1|   |h11inv   h12inv   h12inv||u2|   |u2in1|
        // |v1| = |h21inv   h22inv   h23inv||v2| = |v2in1| / w2in1inv
        // |1 |   |h31inv   h32inv   h33inv||1 |   |  1  |
        // 计算投影归一化坐标
        const float w2in1inv = 1.0/(h31inv*u2+h32inv*v2+h33inv);
        const float u2in1 = (h11inv*u2+h12inv*v2+h13inv)*w2in1inv;
        const float v2in1 = (h21inv*u2+h22inv*v2+h23inv)*w2in1inv;

        // 计算重投影误差 || p1(i) - H12 * p2(i) ||2
        const float squareDist1 = (u1-u2in1)*(u1-u2in1)+(v1-v2in1)*(v1-v2in1);

        const float chiSquare1 = squareDist1*invSigmaSquare;

        // 用阈值标记离群点，内点的话累加得分
        if(chiSquare1>th)
            bIn = false;
        else
            // 误差越大，得分越低
            score += th - chiSquare1;

        // Reprojection error in second image
        // x1in2 = H21*x1
        // // 计算 img1 到 img2 的重投影误差,同上
        const float w1in2inv = 1.0/(h31*u1+h32*v1+h33);
        const float u1in2 = (h11*u1+h12*v1+h13)*w1in2inv;
        const float v1in2 = (h21*u1+h22*v1+h23)*w1in2inv;

        const float squareDist2 = (u2-u1in2)*(u2-u1in2)+(v2-v1in2)*(v2-v1in2);

        const float chiSquare2 = squareDist2*invSigmaSquare;

        if(chiSquare2>th)
            bIn = false;
        else
            score += th - chiSquare2;

        // 如果从 img1 到 img2 和 img2 到 img1 的重投影都满足要求，则标记为内点
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**
 * @brief 对给定的Fundamental矩阵打分
 * 
 * @param[in] F21                       从参考帧到当前帧的基础矩阵
 * @param[in] vbMatchesInliers          匹配好的特征点对的Inliers标记
 * @param[in] sigma                     方差，默认为1
 * @return                              返回得分
 **/
float Initializer::CheckFundamental(const cv::Mat &F21, vector<bool> &vbMatchesInliers, float sigma)
{
    // 利用对极几何原理 p2^T * F * p1 = 0
    // 假设：三维空间中的点 P 在 img1 和 img2 两幅图像上的投影分别为 p1 和 p2
    // 则： p2 一定存在于极线 l2 上，在 直线上的点满足方程 ax + by + c = 0, p2 为像素坐标 [u v 1]^T,
    //      那么根据对极约束的 F * p1 = [a b c]^T 所表示的直线应该就是极线 l2
    //      所以，误差项 e 就可以定义为 p2 到极线 l2 的距离，如果在直线上，则 e = 0
    //      根据点到直线的距离公式： d = (ax + by + c) / sqrt(a * a + b * b)
    //      所以， e = (a * p2.x + b * p2.y + c) / sqrt(a * a + b * b)

    // 算法流程:
    // input: 基础矩阵 F 左右视图匹配点集 mvKeys
    //    do:
    //          for p1(i), p2(i) in mvKeys:
    //              error_i1 = dist_point_to_line(x2, l2)
    //              error_i2 = dist_point_to_line(x1, l1)
    //
    //              w1 = 1 / sigma / sigma
    //              w2 = 1 / sigma / sigma
    //
    //              if error_i1 < th
    //                  score += th - error_i1 * w1
    //              if error_i2 < th
    //                  score += th - error_i2 * w2
    //
    //              if error_i1 > th or error_i2 > th
    //                  p1(i),p2(i) are outer points
    //                  vbMatchesInliers(i) = false
    //              else
    //                  p1(i),p2(i) are inner points
    //                  vbMatchesInliers(i) = true
    //              end
    //          end
    //   output: score, inliers

    // 特征点匹配的个数
    const int N = mvMatches12.size();

    // 提取基础矩阵的各个元素
    const float f11 = F21.at<float>(0,0);
    const float f12 = F21.at<float>(0,1);
    const float f13 = F21.at<float>(0,2);
    const float f21 = F21.at<float>(1,0);
    const float f22 = F21.at<float>(1,1);
    const float f23 = F21.at<float>(1,2);
    const float f31 = F21.at<float>(2,0);
    const float f32 = F21.at<float>(2,1);
    const float f33 = F21.at<float>(2,2);

    vbMatchesInliers.resize(N);

    float score = 0;

    // 自由度为1，显著性水平为0.05对应的临界阈值
    const float th = 3.841;
    // 自由度为2的卡方分布，显著性水平为0.05，对应的阈值
    const float thScore = 5.991;

    const float invSigmaSquare = 1.0/(sigma*sigma);

    // 计算 img1 和 img2 在估计 F 时的得分
    for(int i=0; i<N; i++)
    {
        bool bIn = true;

        // 提取参考帧和当前帧之间匹配的特征点
        const cv::KeyPoint &kp1 = mvKeys1[mvMatches12[i].first];
        const cv::KeyPoint &kp2 = mvKeys2[mvMatches12[i].second];

        const float u1 = kp1.pt.x;
        const float v1 = kp1.pt.y;
        const float u2 = kp2.pt.x;
        const float v2 = kp2.pt.y;

        // Reprojection error in second image
        // l2=F21x1=(a2,b2,c2)
        // 计算 img1 上的点在 img2 上投影得到的极线 l2
        const float a2 = f11*u1+f12*v1+f13;
        const float b2 = f21*u1+f22*v1+f23;
        const float c2 = f31*u1+f32*v1+f33;

        // 计算误差-点到直线距离的平方
        const float num2 = a2*u2+b2*v2+c2;

        const float squareDist1 = num2*num2/(a2*a2+b2*b2);

        // 带权重的误差
        const float chiSquare1 = squareDist1*invSigmaSquare;

        if(chiSquare1>th)
            bIn = false;
        else
            score += thScore - chiSquare1;

        // Reprojection error in second image
        // l1 =x2tF21=(a1,b1,c1)
        // 计算 img2 上的点在 img1 上投影得到的极线 l1
        const float a1 = f11*u2+f21*v2+f31;
        const float b1 = f12*u2+f22*v2+f32;
        const float c1 = f13*u2+f23*v2+f33;

        // 计算误差-点到直线距离的平方
        const float num1 = a1*u1+b1*v1+c1;

        const float squareDist2 = num1*num1/(a1*a1+b1*b1);

        // 带权重的误差
        const float chiSquare2 = squareDist2*invSigmaSquare;

        // ? 判断阈值用的 单自由度的，计算得分用的是双自由度，可能是为了和H矩阵得分统一
        if(chiSquare2>th)
            bIn = false;
        else
            score += thScore - chiSquare2;

        // 如果从 img1 到 img2 和 img2 到 img1 的重投影都满足要求，则标记为内点
        if(bIn)
            vbMatchesInliers[i]=true;
        else
            vbMatchesInliers[i]=false;
    }

    return score;
}

/**
 * @brief 从基础矩阵F中求解位姿R，t及三维点
 * F分解出E，E有四组解，选择计算的有效三维点(在摄像头前方，投影误差小于阈值，视差角大于阈值)最多的作为最优的解
 * @param vbMatchesInliers              匹配点对的内点标记        
 * @param F21                           从参考帧到当前帧的基础矩阵
 * @param K                             相机的内参矩阵
 * @param R21                           计算出来的相机旋转
 * @param t21                           计算出来的相机平移
 * @param vP3D                          世界坐标系下，三角化测量特征点对之后得到的空间坐标
 * @param vbTriangulated                特征点是否成功三角化的标记
 * @param minParallax                   三角测量中，任务测量有效时应满足的最小视差角(视差过小，会引起非常大的误差)，单位是角度
 * @param minTriangulated               为了进行运动恢复，所需要的最少的三角化测量成功的点的个数
 **/
bool Initializer::ReconstructF(vector<bool> &vbMatchesInliers, cv::Mat &F21, cv::Mat &K,
                            cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    // Step 1 统计有效匹配点的个数
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // Step 2 根据基础矩阵和相机内参矩阵求本质矩阵  F = K^-T * E * K^-1
    // Compute Essential Matrix from Fundamental Matrix
    cv::Mat E21 = K.t()*F21*K;

    // 定义本质矩阵的分解结果，形成四组解，分别是
    // (R1, t) (R1, -t) (-R1, t) (-R1, -t)
    cv::Mat R1, R2, t;

    // Recover the 4 motion hypotheses
    // Step 3 从本质矩阵求解两个R解和两个t解，共四组解
    // 由于两个t解互为相反数，这里先只获取一个
    // 虽然这个函数有对t作归一化，但并没有决定整个SLAM过程的尺度
    // 因为 CreateInitialMapMonocular 函数对3D点深度会缩放，反过来影响t
    DecomposeE(E21,R1,R2,t);  

    cv::Mat t1=t;
    cv::Mat t2=-t;

    // Step 4 分别验证求解四种组合，选出最佳组合
    // 原理:若某一组合使恢复得到的3D点位于相机正前方的数量最多，那么该组合就是最佳组合
    // 实现:根据计算的结果组合成四种情况，并依次调用 CheckRT() 进行检查，得到可以进行三角化测量的点的数目

    // Reconstruct with the 4 hyphoteses and check
    // 定义四组解进行三角化测量之后的空间点坐标
    vector<cv::Point3f> vP3D1, vP3D2, vP3D3, vP3D4;

    // 定义四组解标记成功三角化的容器
    vector<bool> vbTriangulated1,vbTriangulated2,vbTriangulated3, vbTriangulated4;

    // 定义四组解中成功三角化的最小的视差角
    float parallax1,parallax2, parallax3, parallax4;

    // 使用同样的匹配点分别检查四组解，记录3D点在摄像头前方且投影误差小于阈值的个数，即为有效3D点个数
    int nGood1 = CheckRT(R1,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D1, 4.0*mSigma2, vbTriangulated1, parallax1);
    int nGood2 = CheckRT(R2,t1,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D2, 4.0*mSigma2, vbTriangulated2, parallax2);
    int nGood3 = CheckRT(R1,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D3, 4.0*mSigma2, vbTriangulated3, parallax3);
    int nGood4 = CheckRT(R2,t2,mvKeys1,mvKeys2,mvMatches12,vbMatchesInliers,K, vP3D4, 4.0*mSigma2, vbTriangulated4, parallax4);

    // 选取最大的可三角化的特征点的数目
    int maxGood = max(nGood1,max(nGood2,max(nGood3,nGood4)));

    // 重置变量
    R21 = cv::Mat();
    t21 = cv::Mat();

    // 确定最小的应该被成功三角化的点数，0.9倍的内点数，最小为50
    int nMinGood = max(static_cast<int>(0.9*N),minTriangulated);

    // 统计四组解中，重建的有效3D点个数 大于 0.7倍的最大三角化特征点数目
    // 如果有多个解满足条件，说明没有区分开最优和次优
    int nsimilar = 0;
    if(nGood1>0.7*maxGood)
        nsimilar++;
    if(nGood2>0.7*maxGood)
        nsimilar++;
    if(nGood3>0.7*maxGood)
        nsimilar++;
    if(nGood4>0.7*maxGood)
        nsimilar++;

    // If there is not a clear winner or not enough triangulated points reject initialization
    // 如果四组解没有足够多的成功三角化的点 或者 没有明显最优的结果，则返回失败
    if(maxGood<nMinGood || nsimilar>1)
    {
        return false;
    }

    // If best reconstruction has enough parallax initialize
    // 选择最佳解记录结果
    if(maxGood==nGood1)
    {
        // 该解下的最小视差角大于函数给定的阈值
        if(parallax1>minParallax)
        {
            // 存储3D坐标点
            vP3D = vP3D1;
            // 获取特征点向量的三角化测量标记
            vbTriangulated = vbTriangulated1;

            // 存储相机姿态
            R1.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood2)
    {
        if(parallax2>minParallax)
        {
            vP3D = vP3D2;
            vbTriangulated = vbTriangulated2;

            R2.copyTo(R21);
            t1.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood3)
    {
        if(parallax3>minParallax)
        {
            vP3D = vP3D3;
            vbTriangulated = vbTriangulated3;

            R1.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }else if(maxGood==nGood4)
    {
        if(parallax4>minParallax)
        {
            vP3D = vP3D4;
            vbTriangulated = vbTriangulated4;

            R2.copyTo(R21);
            t2.copyTo(t21);
            return true;
        }
    }

    return false;
}

/**
 * @brief 用H矩阵恢复R，t和三维空间点
 * H矩阵分解常见右两种方法： Faugeras SVD-based decomposition 和 Zhang SVD-based decomposition
 * 代码使用了 Faugeras SVD-based decomposition，参考文献
 * Motion and structure from motion in a piecewise planar encvironment. International Journal of Pattern Recongnitino and Atificial Intelligence.
 * 
 * @param vbMatchesInliers          匹配点对的内点标记
 * @param H21                       从参考帧到当前帧的单应矩阵
 * @param K                         相机的内参矩阵
 * @param R21                       计算出来的相机旋转
 * @param t21                       计算出来的相机平移
 * @param vP3D                      世界坐标系下，三角化测量特征点对之后得到的空间坐标
 * @param vbTriangulated            特征点是否成功三角化的标记
 * @param minParallax               三角测量中，任务测量有效时应满足的最小视差角(视差过小，会引起非常大的误差)，单位是角度
 * @param minTriangulated           为了进行运动恢复，所需要的最少的三角化测量成功的点的个数
 **/
bool Initializer::ReconstructH(vector<bool> &vbMatchesInliers, cv::Mat &H21, cv::Mat &K,
                      cv::Mat &R21, cv::Mat &t21, vector<cv::Point3f> &vP3D, vector<bool> &vbTriangulated, float minParallax, int minTriangulated)
{
    // 流程：
    //      1.根据H矩阵的奇异值d'=d2 或者d'=-d2 分别计算H矩阵分解平的8组解
    //          1.1 讨论d'>0 时的4组解
    //          1.2 讨论d'<0 时的4组解
    //      2.对8组解进行验证，并选择产生相机前方最多3D点的解作为最优解

    // 统计匹配的特征点对中属于内点(Inliers)的有效点个数
    int N=0;
    for(size_t i=0, iend = vbMatchesInliers.size() ; i<iend; i++)
        if(vbMatchesInliers[i])
            N++;

    // We recover 8 motion hypotheses using the method of Faugeras et al.
    // Motion and structure from motion in a piecewise planar environment.
    // International Journal of Pattern Recognition and Artificial Intelligence, 1988

    // 参考<视觉SLAM十四讲>， H = K * A * K^-1
    // A = K^-1 * H * K
    cv::Mat invK = K.inv();
    cv::Mat A = invK*H21*K;

    // 对矩阵A进行SVD分解
    cv::Mat U,w,Vt,V;
    cv::SVD::compute(A,w,U,Vt,cv::SVD::FULL_UV);
    // 计算Vt的转置
    V=Vt.t();

    // 论文中定义 s = det(U) * det(V)
    float s = cv::determinant(U)*cv::determinant(Vt);

    // 取得各个矩阵的奇异值
    float d1 = w.at<float>(0);
    float d2 = w.at<float>(1);
    float d3 = w.at<float>(2);

    // SVD分解正常的情况下，应该满足 d1 >= d2 >= d3 >= 0
    if(d1/d2<1.00001 || d2/d3<1.00001)
    {
        return false;
    }

    // 定义了8种情况下的8组解
    vector<cv::Mat> vR, vt, vn;
    vR.reserve(8);
    vt.reserve(8);
    vn.reserve(8);

    //n'=[x1 0 x3] 4 posibilities e1=e3=1, e1=1 e3=-1, e1=-1 e3=1, e1=e3=-1
    // 根据论文eq.(12)有
    // x1 = e1 * sqrt((d1 * d1 - d2 * d2)/(d1 * d1 - d3 * d3))
    // x2 = 0
    // x3 = e3 * sqrt((d2 * d2 - d3 * d3)/(d1 * d1 - d3 * d3))
    // 令 aux1 = sqrt((d1 * d1 - d2 * d2)/(d1 * d1 - d3 * d3))
    //   aux3 = sqrt((d2 * d2 - d3 * d3)/(d1 * d1 - d3 * d3))
    // 则 x1 = e1 * aux1
    //   x3 = e3 * aux3
    // e1 和 e3 都可以取 +-1，所以 x1 和 x3 有四种组合
    float aux1 = sqrt((d1*d1-d2*d2)/(d1*d1-d3*d3));
    float aux3 = sqrt((d2*d2-d3*d3)/(d1*d1-d3*d3));
    float x1[] = {aux1,aux1,-aux1,-aux1};
    float x3[] = {aux3,-aux3,aux3,-aux3};

    // 计算当 d' > 0时候的4组解
    //case d'=d2

    // 根据论文eq.(13)有
    // sin(theta) = e1 * e3 * sqrt((d1 * d1 - d2 * d2)*(d2 * d2 - d3 * d3)) / (d1 + d3) / d2
    // cos(theta) = (d2 * d2 + d1 * d3) / (d1 + d3) / d2
    float aux_stheta = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1+d3)*d2);

    float ctheta = (d2*d2+d1*d3)/((d1+d3)*d2);
    float stheta[] = {aux_stheta, -aux_stheta, -aux_stheta, aux_stheta};

    // 计算旋转矩阵R'
    // 根据不同的e1 e3组合计算所得出来的四种R，t的解
    //  |  ctheta    0   -aux_stheta |          |  aux1 |
    //  |    0       1        0      | = R'     |   0   | * (d1 - d3) = t'
    //  |aux_theta   0      ctheta   |          | -aux3 |
    //
    //  |  ctheta    0    aux_stheta |          |  aux1 |
    //  |    0       1        0      | = R'     |   0   | * (d1 - d3) = t'
    //  |-aux_theta  0      ctheta   |          |  aux3 |
    //
    //  |  ctheta    0    aux_stheta |          | -aux1 |
    //  |    0       1        0      | = R'     |   0   | * (d1 - d3) = t'
    //  |-aux_theta  0      ctheta   |          | -aux3 |
    //
    //  |  ctheta    0   -aux_stheta |          | -aux1 |
    //  |    0       1        0      | = R'     |   0   | * (d1 - d3) = t'
    //  |aux_theta   0      ctheta   |          |  aux3 |
    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=ctheta;
        Rp.at<float>(0,2)=-stheta[i];
        Rp.at<float>(2,0)=stheta[i];
        Rp.at<float>(2,2)=ctheta;

        // 根据论文eq(8), R= s * U * R' * Vt
        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        // t = U * t'
        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=-x3[i];
        tp*=d1-d3;

        cv::Mat t = U*tp;
        // ?这里对t的归一化在论文里并没有出现
        vt.push_back(t/cv::norm(t));

        // n = V * n'
        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        // 保持平面法向量向上
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }

    // 计算当 d' < 0时候的4组解
    //case d'=-d2
    float aux_sphi = sqrt((d1*d1-d2*d2)*(d2*d2-d3*d3))/((d1-d3)*d2);

    float cphi = (d1*d3-d2*d2)/((d1-d3)*d2);
    float sphi[] = {aux_sphi, -aux_sphi, -aux_sphi, aux_sphi};

    for(int i=0; i<4; i++)
    {
        cv::Mat Rp=cv::Mat::eye(3,3,CV_32F);
        Rp.at<float>(0,0)=cphi;
        Rp.at<float>(0,2)=sphi[i];
        Rp.at<float>(1,1)=-1;
        Rp.at<float>(2,0)=sphi[i];
        Rp.at<float>(2,2)=-cphi;

        cv::Mat R = s*U*Rp*Vt;
        vR.push_back(R);

        cv::Mat tp(3,1,CV_32F);
        tp.at<float>(0)=x1[i];
        tp.at<float>(1)=0;
        tp.at<float>(2)=x3[i];
        tp*=d1+d3;

        cv::Mat t = U*tp;
        vt.push_back(t/cv::norm(t));

        cv::Mat np(3,1,CV_32F);
        np.at<float>(0)=x1[i];
        np.at<float>(1)=0;
        np.at<float>(2)=x3[i];

        cv::Mat n = V*np;
        if(n.at<float>(2)<0)
            n=-n;
        vn.push_back(n);
    }


    int bestGood = 0;
    int secondBestGood = 0;    
    int bestSolutionIdx = -1;
    float bestParallax = -1;
    vector<cv::Point3f> bestP3D;
    vector<bool> bestTriangulated;

    // Instead of applying the visibility constraints proposed in the Faugeras' paper (which could fail for points seen with low parallax)
    // We reconstruct all hypotheses and check in terms of triangulated points and parallax
    
    // 对8组解进行验证，并选择产生相机前方3D点最多的解作为最优解
    for(size_t i=0; i<8; i++)
    {
        // 第i组解对应的最小的视差角
        float parallaxi;
        // 三角化测量之后的特征点的空间坐标
        vector<cv::Point3f> vP3Di;
        // 标记特征点是否被三角化
        vector<bool> vbTriangulatedi;

        // 这里调用 Initializer::CheckRT 计算good点的数目
        int nGood = CheckRT(
            vR[i],vt[i],
            mvKeys1,mvKeys2,
            mvMatches12,
            vbMatchesInliers,
            K,
            vP3Di, 
            4.0*mSigma2, 
            vbTriangulatedi, 
            parallaxi);

        // 更新历史最优和次优的解
        if(nGood>bestGood)
        {
            // 如果当前组解的Good点数是历史最优，那么之前的历史最优就成了历史次优
            secondBestGood = bestGood;
            // 更新历史最优解
            bestGood = nGood;
            // 最优解的组索引(就是当前次遍历)
            bestSolutionIdx = i;
            // 更新变量
            bestParallax = parallaxi;
            bestP3D = vP3Di;
            bestTriangulated = vbTriangulatedi;
        }
        else if(nGood>secondBestGood)
        {
            // 小于历史最优，但是大于历史次优
            secondBestGood = nGood;
        }
    }

    // 选择最优解，要满足下面四个条件
    // 1. Good点数目最优解明显大于次优解,这里取0.75的经验值
    // 2. 视差角大于规定的阈值
    // 3. Good点数大于规定的最少的被三角化的点数
    // 4. Good点数要足够多，达到总数的90%以上
    if(secondBestGood<0.75*bestGood && 
        bestParallax>=minParallax && 
        bestGood>minTriangulated && 
        bestGood>0.9*N)
    {
        // 从最优解的索引访问R，t
        vR[bestSolutionIdx].copyTo(R21);
        vt[bestSolutionIdx].copyTo(t21);
        // 最优解情况下所有特征点的三维空间点
        vP3D = bestP3D;
        // 获取特征点满足最小视差角要求的 标记
        vbTriangulated = bestTriangulated;

        // 返回True，找到了最优解
        return true;
    }

    return false;
}

/**
 * @brief 三角化测量计算空间点坐标
 * 
 * @param kp1       参考帧的特征点
 * @param kp2       当前帧的特征点
 * @param P1        相机1(参考帧)的投影矩阵
 * @param P2        相机2(当前帧)的投影矩阵
 * @param x3D       存储计算出来的空间点坐标
 **/
void Initializer::Triangulate(const cv::KeyPoint &kp1, const cv::KeyPoint &kp2, const cv::Mat &P1, const cv::Mat &P2, cv::Mat &x3D)
{
    // 原理:
    // Triangulation: 已知匹配的特征点对x,x' 和各自相机矩阵P,P', 估计三维点 X
    // x' = P' * X,  x = P * X
    // 他们都属于 x = a * P * X 模型，a是深度系数
    //                          |X|
    // |x|     |p1 p2  p3  p4 | |Y|      |x|     |--p0--|   |.|
    // |y| = a |p5 p6  p7  p8 |*|Z| ===> |y| = a |--p1--| * |X|
    // |z|     |p9 p10 p11 p12| |1|      |z|     |--p2--|   |.|
    // 采用DLT的方法，等式两边同时乘上 x^(反对称矩阵)
    // | -p1+y*p2 |
    // |  p0-x*p2 | * X = 0
    // |-y*p0+x*p1|
    // 有两组点，分别取等式前两行
    // |-p1  + y * p2 |       |0|
    // | p0  - x * p2 |       |0|
    // |-p1' + y'* p2'| * X = |0|
    // | p0' - x'* p2'|       |0|
    // 经整理，就得到程序中使用的式子
    // |x * p2 - p0 |       |0|
    // |y * p2 - p1 |       |0|
    // |x'* p2'- p0'| * X = |0|
    // |y'* p2'- p1'|       |0|
    // 计算左边矩阵的SVD，取右奇异向量的最后一行即为解

    cv::Mat A(4,4,CV_32F);

    // 构造如上原理所述的矩阵A
    A.row(0) = kp1.pt.x*P1.row(2)-P1.row(0);
    A.row(1) = kp1.pt.y*P1.row(2)-P1.row(1);
    A.row(2) = kp2.pt.x*P2.row(2)-P2.row(0);
    A.row(3) = kp2.pt.y*P2.row(2)-P2.row(1);

    // 对A进行SVD分解，并取右奇异向量的最后一行
    cv::Mat u,w,vt;
    cv::SVD::compute(A,w,u,vt,cv::SVD::MODIFY_A| cv::SVD::FULL_UV);
    x3D = vt.row(3).t();
    // 将空间点坐标化为齐次形式，[X Y Z 1]'
    x3D = x3D.rowRange(0,3)/x3D.at<float>(3);
}

/**
 * @brief 归一化特征点到同一尺度，作为后续normalize DLT的输入
 * [x' y' 1]' = T * [x y 1]'
 * 归一化后的x',y'的均值为0，sum(abs(x_i'-0))=1, sum(abs(y_i'-0))=1
 * 
 * 为什么要归一化？
 * 在相似变换之后(点在不同的坐标系下)，他们的单应矩阵是不相同的
 * 如果图像存在噪声，使得点的坐标发生了变化，那么它的单应矩阵也会发生变化
 * 我们采取的方法是将点的坐标放到同一坐标系下，并将缩放尺度也进行统一
 * 对同一幅图像的坐标进行相同的变换，不同图像进行不同变换
 * 缩放尺度是为了让噪声对于图像的影响在一个数量级上
 * 
 * Step 1 计算特征点X，Y坐标的均值
 * Step 2 计算特征点X，Y坐标离均值的平均偏离程度
 * Step 3 将X坐标和Y坐标分别进行尺度归一化，使得X坐标和Y坐标的一阶绝对矩分别为1
 * Step 4 计算归一化矩阵，其实就是前面做的操作用矩阵变换来表示而已
 * 
 * @param [in] vKeys                        待归一化的特征点
 * @param [in&out] vNormalizedPoints        特征点归一化后的坐标
 * @param [in&out] T                        归一化特征点的矩阵
 **/
void Initializer::Normalize(const vector<cv::KeyPoint> &vKeys, vector<cv::Point2f> &vNormalizedPoints, cv::Mat &T)
{
    // Step 1 计算特征点X，Y坐标的均值 meanX,meanY
    float meanX = 0;
    float meanY = 0;
    const int N = vKeys.size();

    vNormalizedPoints.resize(N);

    for(int i=0; i<N; i++)
    {
        meanX += vKeys[i].pt.x;
        meanY += vKeys[i].pt.y;
    }

    meanX = meanX/N;
    meanY = meanY/N;

    // Step 2 计算特征点X，Y坐标离均值的平均偏离程度 meanDevX 和 meanDevY
    float meanDevX = 0;
    float meanDevY = 0;

    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vKeys[i].pt.x - meanX;
        vNormalizedPoints[i].y = vKeys[i].pt.y - meanY;

        meanDevX += fabs(vNormalizedPoints[i].x);
        meanDevY += fabs(vNormalizedPoints[i].y);
    }

    // 取平均偏离程度的倒数作为一个缩放因子
    meanDevX = meanDevX/N;
    meanDevY = meanDevY/N;

    float sX = 1.0/meanDevX;
    float sY = 1.0/meanDevY;

    // Step 3 将X坐标和Y坐标分别进行尺度归一化，使得X坐标和Y坐标的一阶绝对矩为1
    for(int i=0; i<N; i++)
    {
        vNormalizedPoints[i].x = vNormalizedPoints[i].x * sX;
        vNormalizedPoints[i].y = vNormalizedPoints[i].y * sY;
    }

    // Step 4 计算归一化矩阵，就是把上述的过程用矩阵表示
    // |sX  0   -meanx*sX |
    // |0   sY  -meany*sY |
    // |0   0        1    |
    T = cv::Mat::eye(3,3,CV_32F);
    T.at<float>(0,0) = sX;
    T.at<float>(1,1) = sY;
    T.at<float>(0,2) = -meanX*sX;
    T.at<float>(1,2) = -meanY*sY;
    // 计算完反推的时候用
}

/**
 * @brief 用R，t来对匹配的特征点三角化测量，并根据测量的结果判断R，t的合理性
 * 
 * @param R                 旋转矩阵    
 * @param t                 平移向量
 * @param vKeys1            参考帧特征点
 * @param vKeys2            当前帧特征点
 * @param vMatches12        两帧的匹配信息
 * @param vbMatchesInliers  特征点对内点标记
 * @param K                 相机内参矩阵
 * @param vP3D              三角测量之后的空间点坐标
 * @param th2               重投影误差的阈值
 * @param vbGood            标记成功三角化的点
 * @param parallax          计算出来的最小的视差角
 **/
int Initializer::CheckRT(const cv::Mat &R, const cv::Mat &t, const vector<cv::KeyPoint> &vKeys1, const vector<cv::KeyPoint> &vKeys2,
                       const vector<Match> &vMatches12, vector<bool> &vbMatchesInliers,
                       const cv::Mat &K, vector<cv::Point3f> &vP3D, float th2, vector<bool> &vbGood, float &parallax)
{
    // 取出相机的内参
    // Calibration parameters
    const float fx = K.at<float>(0,0);
    const float fy = K.at<float>(1,1);
    const float cx = K.at<float>(0,2);
    const float cy = K.at<float>(1,2);

    // 初始化标记成功三角化的向量和空间做坐标的向量
    vbGood = vector<bool>(vKeys1.size(),false);
    vP3D.resize(vKeys1.size());

    // 存储计算出来的每组点的视差
    vector<float> vCosParallax;
    vCosParallax.reserve(vKeys1.size());

    // Camera 1 Projection Matrix K[I|0]
    // Step 1 计算相机的投影矩阵
    // 投影矩阵P是一个3X4的矩阵，可以将空间中的一个点投影到平面上，获得其平面坐标(齐次坐标)
    // 对于第一个相机是 P1 = K[I|0]
    cv::Mat P1(3,4,             // 矩阵大小3X4
               CV_32F,          // 包含的数据类型
               cv::Scalar(0));  // 初始值均为0
    // 将K矩阵直接复制到P1矩阵的前三行前三列，因为 K * I = K
    K.copyTo(P1.rowRange(0,3).colRange(0,3));
    // 第一个相机的光心设置为世界坐标系的原点
    cv::Mat O1 = cv::Mat::zeros(3,1,CV_32F);

    // Camera 2 Projection Matrix K[R|t]
    // 计算第二个相机的投影矩阵
    cv::Mat P2(3,4,CV_32F);
    R.copyTo(P2.rowRange(0,3).colRange(0,3));
    t.copyTo(P2.rowRange(0,3).col(3));
    P2 = K*P2;

    // 第二个相机的光心在世界坐标系下的坐标
    cv::Mat O2 = -R.t()*t;
    // 在遍历开始前，Good点数量清零
    int nGood=0;

    // 开始遍历
    for(size_t i=0, iend=vMatches12.size();i<iend;i++)
    {
        if(!vbMatchesInliers[i])
            continue;

        // Step 2 获取匹配的特征点,调用Triangulate进行三角化，得到三角化之后的3D点坐标
        // kp1 kp2 是匹配好的特征点
        const cv::KeyPoint &kp1 = vKeys1[vMatches12[i].first];
        const cv::KeyPoint &kp2 = vKeys2[vMatches12[i].second];
        cv::Mat p3dC1;

        // 利用三角化恢复三维点
        Triangulate(kp1,kp2,P1,P2,p3dC1);

        // Step 3 检查三角化的三维点坐标是否合理(非无穷值)
        if(!isfinite(p3dC1.at<float>(0)) || !isfinite(p3dC1.at<float>(1)) || !isfinite(p3dC1.at<float>(2)))
        {
            vbGood[vMatches12[i].first]=false;
            continue;
        }

        // Check parallax
        // Step 4 通过三维点深度值的正负、两相机光心视差角大小检查是否合理
        // 计算从相机1的光心 指向 空间点的向量，dist是模长
        cv::Mat normal1 = p3dC1 - O1;
        float dist1 = cv::norm(normal1);

        // 计算从相机2的光心 指向 空间点的向量
        cv::Mat normal2 = p3dC1 - O2;
        float dist2 = cv::norm(normal2);

        // 根据余弦公式求夹角
        float cosParallax = normal1.dot(normal2)/(dist1*dist2);

        // 检查相机1下的空间点深度信息
        // 如果深度值为负，为不合理的三维点
        // Check depth in front of first camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        // ? 如果余弦值小于0.99998 且 深度值为负，跳过。余弦值小于0.99998意味着夹角大于某一个很小的阈值
        if(p3dC1.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // 检查相机2下的空间点深度信息
        // Check depth in front of second camera (only if enough parallax, as "infinite" points can easily go to negative depth)
        cv::Mat p3dC2 = R*p3dC1+t;

        if(p3dC2.at<float>(2)<=0 && cosParallax<0.99998)
            continue;

        // Step 5 判断空间点在两帧上的重投影误差，大于阈值则舍弃
        // 判断在第一个图像上的投影误差
        // Check reprojection error in first image
        // 图像1像素坐标系上的x，y值
        float im1x, im1y;
        // 空间点的深度值的倒数，1/Z
        float invZ1 = 1.0/p3dC1.at<float>(2);
        // 计算图像1的像素坐标
        im1x = fx*p3dC1.at<float>(0)*invZ1+cx;
        im1y = fy*p3dC1.at<float>(1)*invZ1+cy;

        // 参考帧上的重投影误差
        float squareError1 = (im1x-kp1.pt.x)*(im1x-kp1.pt.x)+(im1y-kp1.pt.y)*(im1y-kp1.pt.y);

        if(squareError1>th2)
            continue;

        // 判断在第二个图像上的投影误差
        // Check reprojection error in second image
        // 同上
        float im2x, im2y;
        float invZ2 = 1.0/p3dC2.at<float>(2);
        im2x = fx*p3dC2.at<float>(0)*invZ2+cx;
        im2y = fy*p3dC2.at<float>(1)*invZ2+cy;

        float squareError2 = (im2x-kp2.pt.x)*(im2x-kp2.pt.x)+(im2y-kp2.pt.y)*(im2y-kp2.pt.y);

        if(squareError2>th2)
            continue;

        // Step 6 统计经过检验的3D点个数，记录3D点视差角
        vCosParallax.push_back(cosParallax);
        // 存储这个空间点以参考帧作为世界坐标系的坐标
        vP3D[vMatches12[i].first] = cv::Point3f(p3dC1.at<float>(0),p3dC1.at<float>(1),p3dC1.at<float>(2));
        // Good点计数
        nGood++;

        // ? Good点标志,在这里会让计数和标志数不太一样，判断条件是要大于一点点角度
        if(cosParallax<0.99998)
            vbGood[vMatches12[i].first]=true;
    }

    // Step 7 得到3D点中较小的视差角，并转化为角度值
    if(nGood>0)
    {
        // 将余弦值按照从小到大排序
        sort(vCosParallax.begin(),vCosParallax.end());

        // 在50和成功三角化的点的数量中取一个较小者
        // ? 取得是一个较大的余弦值，对应较小的视差角
        size_t idx = min(50,int(vCosParallax.size()-1));
        parallax = acos(vCosParallax[idx])*180/CV_PI;
    }
    else
        parallax=0;

    return nGood;
}

/**
 * @brief 分解Essential矩阵得到R，t
 * 分解E矩阵将得到四组解，这4组解分别为[R1, t] [R1, -t] [-R1, t] [-R1, -t]
 * 
 * @param E         本质矩阵
 * @param R1        旋转矩阵1
 * @param R2        旋转矩阵2
 * @param t         平移向量，另一个取相反数
 **/
void Initializer::DecomposeE(const cv::Mat &E, cv::Mat &R1, cv::Mat &R2, cv::Mat &t)
{
    cv::Mat u,w,vt;
    // 对本质矩阵SVD分解
    cv::SVD::compute(E,w,u,vt);

    // 左奇异向量的最后一行就是t，并对其进行归一化
    // Reference: <Multiple View in Computer Vision, 2nd Edition>, P259
    u.col(2).copyTo(t);
    t=t/cv::norm(t);

    // 构造一个绕Z轴旋转pi/2的矩阵W
    cv::Mat W(3,3,CV_32F,cv::Scalar(0));
    W.at<float>(0,1)=-1;
    W.at<float>(1,0)=1;
    W.at<float>(2,2)=1;

    // 只保留行列式为正的结果
    R1 = u*W*vt;
    if(cv::determinant(R1)<0)
        R1=-R1;

    R2 = u*W.t()*vt;
    if(cv::determinant(R2)<0)
        R2=-R2;
}

} //namespace ORB_SLAM
