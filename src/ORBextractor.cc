/**
* This file is part of ORB-SLAM2.
* This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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
/**
* Software License Agreement (BSD License)
*
*  Copyright (c) 2009, Willow Garage, Inc.
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions
*  are met:
*
*   * Redistributions of source code must retain the above copyright
*     notice, this list of conditions and the following disclaimer.
*   * Redistributions in binary form must reproduce the above
*     copyright notice, this list of conditions and the following
*     disclaimer in the documentation and/or other materials provided
*     with the distribution.
*   * Neither the name of the Willow Garage nor the names of its
*     contributors may be used to endorse or promote products derived
*     from this software without specific prior written permission.
*
*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
*  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
*  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
*  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
*  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
*  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
*  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
*  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
*  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
*  POSSIBILITY OF SUCH DAMAGE.
*
*/


#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

#include "ORBextractor.h"


using namespace cv;
using namespace std;

namespace ORB_SLAM2
{
const int PATCH_SIZE = 31;  // 圆的直径
const int HALF_PATCH_SIZE = 15; // 圆的半径，除去了原点
const int EDGE_THRESHOLD = 19;  // 创建图像金字塔时，图像预留的边界大小


// 计算特征点的方向，返回值单位是°，精度是0.3°
static float IC_Angle(const Mat& image, 
                        Point2f pt,         // 中心点的坐标  
                        const vector<int> & u_max)
{
    int m_01 = 0,   // y坐标的矩 
        m_10 = 0;   // x坐标的矩

    // center[0]对应中心点的像素值，center[u+v*step]代表向下v行,向右u列的点的像素值
    const uchar* center = &image.at<uchar> (cvRound(pt.y), cvRound(pt.x));

    // Treat the center line differently, v=0
    // 先计算水平经过圆心的线
    for (int u = -HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; ++u)
        m_10 += u * center[u];

    // Go line by line in the circular patch
    // step1是图像一行包含的字节总数
    int step = (int)image.step1();
    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        // 对称地画上下两条线，计算m10和m01
        int v_sum = 0;
        int d = u_max[v];
        for (int u = -d; u <= d; ++u)
        {
            // val_plus是中心线下方的点的像素值
            // val_minus是中心线上方的点的像素值
            // 两个点关于中心线上下对称，所以v_sum可以一起计算，只需要遍历上半圆即可
            int val_plus = center[u + v*step], val_minus = center[u - v*step];
            v_sum += (val_plus - val_minus);
            m_10 += u * (val_plus + val_minus);
        }
        m_01 += v * v_sum;
    }

    return fastAtan2((float)m_01, (float)m_10);
}

// 用于把角度转化为弧度的乘数因子
const float factorPI = (float)(CV_PI/180.f);

/**
 * @brief 计算ORB特征点的描述子，是一个静态函数，只能在本文件中调用
 * 
 * @param[in] kpt       特征点对象
 * @param[in] img       特征点所在的图像
 * @param[in] pattern   预定义好的随即采样点集
 * @param[out] desc     保存描述子，长度为32*8bit
 **/
static void computeOrbDescriptor(const KeyPoint& kpt,
                                 const Mat& img, const Point* pattern,
                                 uchar* desc)
{
    // Step 1 把pattern图像按照关键点的方向进行旋转
    // 把角度转化为弧度
    float angle = (float)kpt.angle*factorPI;
    // 计算旋转角度的余弦值和正弦值
    float a = (float)cos(angle), b = (float)sin(angle);

    // 获取图像中心的指针
    const uchar* center = &img.at<uchar>(cvRound(kpt.pt.y), cvRound(kpt.pt.x));
    const int step = (int)img.step;

    // 这里是将pattern里每个点的向量进行旋转，旋转向量表示如下
    // |--        --|
    // |    a   -b  |
    // |    b    a  |
    // |--        --|
    // x'=xa-yb,  y'=xb+ya
    // 定义一个函数用于获取旋转后pattern点对在当前坐标系下的坐标
    #define GET_VALUE(idx) \
        center[cvRound(pattern[idx].x*b + pattern[idx].y*a)*step + \
               cvRound(pattern[idx].x*a - pattern[idx].y*b)]

    // brief描述子由32*8位组成
    // 其中每一位来自两个像素点的值的比较，所以每比较出8bit的结果，需要16个随机点，这就是pattern+=16的原因
    for (int i = 0; i < 32; ++i, pattern += 16)
    {
        int t0,     // 参与比较的一个特征点的灰度值 
            t1,     // 参与比较的另一个特征点的灰度值
            val;    // 描述子这个字节的比较结果
        t0 = GET_VALUE(0); t1 = GET_VALUE(1);
        val = t0 < t1;
        t0 = GET_VALUE(2); t1 = GET_VALUE(3);
        val |= (t0 < t1) << 1;
        t0 = GET_VALUE(4); t1 = GET_VALUE(5);
        val |= (t0 < t1) << 2;
        t0 = GET_VALUE(6); t1 = GET_VALUE(7);
        val |= (t0 < t1) << 3;
        t0 = GET_VALUE(8); t1 = GET_VALUE(9);
        val |= (t0 < t1) << 4;
        t0 = GET_VALUE(10); t1 = GET_VALUE(11);
        val |= (t0 < t1) << 5;
        t0 = GET_VALUE(12); t1 = GET_VALUE(13);
        val |= (t0 < t1) << 6;
        t0 = GET_VALUE(14); t1 = GET_VALUE(15);
        val |= (t0 < t1) << 7;

        // 将8位的计算结果存进desc
        desc[i] = (uchar)val;
    }

    // 为了避免和其他程序冲突，在函数用完之后，就取消宏定义
    #undef GET_VALUE
}


static int bit_pattern_31_[256*4] =
{
    8,-3, 9,5/*mean (0), correlation (0)*/,
    4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
    -11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
    7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
    2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
    1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
    -2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
    -13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
    -13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
    10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
    -13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
    -11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
    7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
    -4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
    -13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
    -9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
    12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
    -3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
    -6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
    11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
    4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
    5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
    3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
    -8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
    -2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
    -13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
    -7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
    -4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
    -10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
    5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
    5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
    1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
    9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
    4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
    2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
    -4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
    -8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
    4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
    0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
    -13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
    -3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
    -6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
    8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
    0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
    7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
    -13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
    10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
    -6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
    10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
    -13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
    -13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
    3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
    5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
    -1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
    3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
    2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
    -13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
    -13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
    -13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
    -7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
    6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
    -9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
    -2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
    -12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
    3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
    -7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
    -3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
    2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
    -11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
    -1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
    5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
    -4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
    -9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
    -12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
    10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
    7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
    -7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
    -4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
    7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
    -7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
    -13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
    -3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
    7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
    -13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
    1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
    2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
    -4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
    -1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
    7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
    1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
    9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
    -1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
    -13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
    7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
    12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
    6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
    5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
    2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
    3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
    2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
    9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
    -8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
    -11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
    1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
    6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
    2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
    6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
    3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
    7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
    -11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
    -10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
    -5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
    -10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
    8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
    4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
    -10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
    4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
    -2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
    -5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
    7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
    -9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
    -5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
    8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
    -9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
    1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
    7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
    -2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
    11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
    -12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
    3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
    5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
    0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
    -9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
    0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
    -1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
    5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
    3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
    -13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
    -5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
    -4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
    6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
    -7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
    -13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
    1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
    4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
    -2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
    2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
    -2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
    4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
    -6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
    -3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
    7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
    4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
    -13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
    7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
    7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
    -7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
    -8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
    -13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
    2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
    10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
    -6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
    8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
    2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
    -11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
    -12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
    -11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
    5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
    -2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
    -1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
    -13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
    -10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
    -3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
    2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
    -9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
    -4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
    -4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
    -6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
    6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
    -13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
    11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
    7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
    -1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
    -4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
    -7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
    -13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
    -7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
    -8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
    -5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
    -13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
    1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
    1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
    9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
    5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
    -1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
    -9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
    -1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
    -13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
    8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
    2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
    7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
    -10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
    -10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
    4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
    3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
    -4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
    5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
    4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
    -9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
    0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
    -12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
    3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
    -10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
    8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
    -8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
    2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
    10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
    6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
    -7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
    -3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
    -1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
    -3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
    -8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
    4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
    2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
    6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
    3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
    11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
    -3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
    4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
    2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
    -10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
    -13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
    -13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
    6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
    0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
    -13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
    -9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
    -13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
    5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
    2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
    -1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
    9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
    11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
    3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
    -1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
    3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
    -13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
    5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
    8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
    7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
    -10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
    7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
    9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
    7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
    -1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
};

ORBextractor::ORBextractor( int _nfeatures,     // 需要提取的特征点数量
                            float _scaleFactor, // 图像金字塔每层之间的缩放倍数
                            int _nlevels,       // 金字塔的层数
                            int _iniThFAST,     // 初始化的FAST corner阈值，提取较明显的特征点
                            int _minThFAST      // 如果纹理不明显，则采用这个，如果还是不够理想，可以减小该值
                            ):
    nfeatures(_nfeatures), scaleFactor(_scaleFactor), nlevels(_nlevels),
    iniThFAST(_iniThFAST), minThFAST(_minThFAST)
{
    mvScaleFactor.resize(nlevels);
    mvLevelSigma2.resize(nlevels);
    mvScaleFactor[0]=1.0f;
    mvLevelSigma2[0]=1.0f;
    for(int i=1; i<nlevels; i++)
    {
        mvScaleFactor[i]=mvScaleFactor[i-1]*scaleFactor;
        // Sigma2 = 缩放倍数的平方
        mvLevelSigma2[i]=mvScaleFactor[i]*mvScaleFactor[i];
    }

    mvInvScaleFactor.resize(nlevels);
    mvInvLevelSigma2.resize(nlevels);
    for(int i=0; i<nlevels; i++)
    {
        mvInvScaleFactor[i]=1.0f/mvScaleFactor[i];
        mvInvLevelSigma2[i]=1.0f/mvLevelSigma2[i];
    }

    mvImagePyramid.resize(nlevels);

    // 每一层的特征点数量
    mnFeaturesPerLevel.resize(nlevels);
    float factor = 1.0f / scaleFactor;
    // 按每一层图像的长度均摊，而不是面积均摊(推导时用面积更容易理解)
    float nDesiredFeaturesPerScale = nfeatures*(1 - factor)/(1 - (float)pow((double)factor, (double)nlevels));

    int sumFeatures = 0;
    for( int level = 0; level < nlevels-1; level++ )
    {
        mnFeaturesPerLevel[level] = cvRound(nDesiredFeaturesPerScale);
        sumFeatures += mnFeaturesPerLevel[level];
        nDesiredFeaturesPerScale *= factor;
    }
    // 为保证特征点的总数一致，最上一层待提取的特征点数量单独计算
    mnFeaturesPerLevel[nlevels-1] = std::max(nfeatures - sumFeatures, 0);

    // 成员变量 pattern 的长度，也就是点的个数，这里的512也就代表着512个点(上面的数组存储的是坐标，所以是 256*2*2)
    const int npoints = 512;
    const Point* pattern0 = (const Point*)bit_pattern_31_;
    std::copy(pattern0, pattern0 + npoints, std::back_inserter(pattern));

    //This is for orientation
    // pre-compute the end of a row in a circular patch
    // 计算圆形边界，u对应横坐标，v对应纵坐标。
    // 在直角坐标系内，水平画直线与圆相交，交点的值就是umax
    // HALF_PATCH_SIZE 对应圆的半径，存储不同的v所对应的umax
    umax.resize(HALF_PATCH_SIZE + 1);

    int v, v0, vmax = cvFloor(HALF_PATCH_SIZE * sqrt(2.f) / 2 + 1);
    int vmin = cvCeil(HALF_PATCH_SIZE * sqrt(2.f) / 2);
    const double hp2 = HALF_PATCH_SIZE*HALF_PATCH_SIZE;
    // 利用三角形来求umax
    for (v = 0; v <= vmax; ++v)
        umax[v] = cvRound(sqrt(hp2 - v * v));

    // Make sure we are symmetric
    // 利用圆几何对称的性质来求剩下的umax，避免因cvRound带来区域的不对称
    for (v = HALF_PATCH_SIZE, v0 = 0; v >= vmin; --v)
    {
        while (umax[v0] == umax[v0 + 1])
            ++v0;
        umax[v] = v0;
        ++v0;
    }

}

static void computeOrientation(const Mat& image,                // 输入的图像
                                vector<KeyPoint>& keypoints,    // 存储的每一层的关键点
                                const vector<int>& umax)        // 每一横线与圆交点对应的横坐标的值(最大值)
{
    for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
         keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
    {
        keypoint->angle = IC_Angle(image, keypoint->pt, umax);
    }
}

// 将一个根节点分为四个子节点，把根节点内关键点按几何位置分给四个子节点
// 根节点的关键点内容并没有被清除
void ExtractorNode::DivideNode(ExtractorNode &n1, ExtractorNode &n2, ExtractorNode &n3, ExtractorNode &n4)
{
    const int halfX = ceil(static_cast<float>(UR.x-UL.x)/2);
    const int halfY = ceil(static_cast<float>(BR.y-UL.y)/2);

    //Define boundaries of childs
    n1.UL = UL;
    n1.UR = cv::Point2i(UL.x+halfX,UL.y);
    n1.BL = cv::Point2i(UL.x,UL.y+halfY);
    n1.BR = cv::Point2i(UL.x+halfX,UL.y+halfY);
    n1.vKeys.reserve(vKeys.size());

    n2.UL = n1.UR;
    n2.UR = UR;
    n2.BL = n1.BR;
    n2.BR = cv::Point2i(UR.x,UL.y+halfY);
    n2.vKeys.reserve(vKeys.size());

    n3.UL = n1.BL;
    n3.UR = n1.BR;
    n3.BL = BL;
    n3.BR = cv::Point2i(n1.BR.x,BL.y);
    n3.vKeys.reserve(vKeys.size());

    n4.UL = n3.UR;
    n4.UR = n2.BR;
    n4.BL = n3.BR;
    n4.BR = BR;
    n4.vKeys.reserve(vKeys.size());

    //Associate points to childs
    for(size_t i=0;i<vKeys.size();i++)
    {
        const cv::KeyPoint &kp = vKeys[i];
        if(kp.pt.x<n1.UR.x)
        {
            if(kp.pt.y<n1.BR.y)
                n1.vKeys.push_back(kp);
            else
                n3.vKeys.push_back(kp);
        }
        else if(kp.pt.y<n1.BR.y)
            n2.vKeys.push_back(kp);
        else
            n4.vKeys.push_back(kp);
    }

    // 对只有一个关键点的子节点，设置为不再划分
    if(n1.vKeys.size()==1)
        n1.bNoMore = true;
    if(n2.vKeys.size()==1)
        n2.bNoMore = true;
    if(n3.vKeys.size()==1)
        n3.bNoMore = true;
    if(n4.vKeys.size()==1)
        n4.bNoMore = true;

}

// 使用四叉树法对特征点进行平均分配，写的是OctTree，实际上是四叉树
vector<cv::KeyPoint> ORBextractor::DistributeOctTree(   // 返回值是一个保存右特征点的vecotr容器
    const vector<cv::KeyPoint>& vToDistributeKeys,      // 等待进行分配的特征点
    const int &minX,
    const int &maxX, 
    const int &minY, 
    const int &maxY, 
    const int &N,                                       // 希望提取的特征点数量
    const int &level)                                   // 指定的层数
{
    // Compute how many initial nodes
    // 计算应该生成的初始节点个数，根节点的数量nIni是图像的宽高比，一般是1或2   
    const int nIni = round(static_cast<float>(maxX-minX)/(maxY-minY));

    // 初始节点x方向有多少个像素
    const float hX = static_cast<float>(maxX-minX)/nIni;

    list<ExtractorNode> lNodes;

    vector<ExtractorNode*> vpIniNodes;
    vpIniNodes.resize(nIni);

    for(int i=0; i<nIni; i++)
    {
        // 生成一个提取器节点
        ExtractorNode ni;

        // 设置提取器节点的图像边界
        ni.UL = cv::Point2i(hX*static_cast<float>(i),0);    // UpLeft
        ni.UR = cv::Point2i(hX*static_cast<float>(i+1),0);  // UpRight
        ni.BL = cv::Point2i(ni.UL.x,maxY-minY);             // BottomLeft
        ni.BR = cv::Point2i(ni.UR.x,maxY-minY);             // BottomRight
        // 重设vKeys的大小
        ni.vKeys.reserve(vToDistributeKeys.size());

        lNodes.push_back(ni);
        vpIniNodes[i] = &lNodes.back();
    }

    //Associate points to childs
    // 将特征点分配到子提取器节点中

    for(size_t i=0;i<vToDistributeKeys.size();i++)
    {
        const cv::KeyPoint &kp = vToDistributeKeys[i];
        // 根据关键点的x坐标所在的位置划分所属提取器
        vpIniNodes[kp.pt.x/hX]->vKeys.push_back(kp);
    }

    list<ExtractorNode>::iterator lit = lNodes.begin();

    // 对根节点做一个预处理
    while(lit!=lNodes.end())
    {
        if(lit->vKeys.size()==1)
        {
            // 只有一个关键点，不再继续划分
            lit->bNoMore=true;
            lit++;
        }
        else if(lit->vKeys.empty())
            // 该提取器没有关键点，直接清除掉
            // 该情况下，函数返回了指向下一个元素的指针(当前元素已被清楚)，不需要执行++
            lit = lNodes.erase(lit);
        else
            lit++;
    }

    bool bFinish = false;

    int iteration = 0;

    // 节点包含的关键点个数和节点本身的指针，该vector容器只包含可以被分裂的节点，在快分裂结束时才被用到，lNode包含了所有含有关键点的节点
    vector<pair<int,ExtractorNode*> > vSizeAndPointerToNode;
    // 按根节点的四倍预留，四叉树
    vSizeAndPointerToNode.reserve(lNodes.size()*4);

    while(!bFinish)
    {
        iteration++;

        int prevSize = lNodes.size();

        lit = lNodes.begin();

        int nToExpand = 0;

        vSizeAndPointerToNode.clear();

        // 对当前所有节点的一次遍历
        while(lit!=lNodes.end())
        {
            if(lit->bNoMore)
            {
                // If node only contains one point do not subdivide and continue
                lit++;
                continue;
            }
            else
            {
                // If more than one point, subdivide
                ExtractorNode n1,n2,n3,n4;
                lit->DivideNode(n1,n2,n3,n4);

                // Add childs if they contain points
                if(n1.vKeys.size()>0)
                {
                    lNodes.push_front(n1);  
                    // 在根节点分为4个字节点的时候，就已经对大小为1的情况进行了处理，设置NoMore=1                  
                    if(n1.vKeys.size()>1)
                    {
                        // 可以被分裂的子节点数量+1
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                        // 用自己的成员变量lit记录自己在这个列表里的位置
                        // lit是一个iterator，便于分裂时从列表中删除掉自己
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n2.vKeys.size()>0)
                {
                    lNodes.push_front(n2);
                    if(n2.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n3.vKeys.size()>0)
                {
                    lNodes.push_front(n3);
                    if(n3.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }
                if(n4.vKeys.size()>0)
                {
                    lNodes.push_front(n4);
                    if(n4.vKeys.size()>1)
                    {
                        nToExpand++;
                        vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                        lNodes.front().lit = lNodes.begin();
                    }
                }

                // 此时lit应该指向这四个子节点的根节点，并把它从list中清除
                lit=lNodes.erase(lit);
                continue;
            }
        }       

        // Finish if there are more nodes than required features
        // or all nodes contain just one point
        if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
        {
            bFinish = true;
        }
        else if(((int)lNodes.size()+nToExpand*3)>N)     // 加3——删除根节点，加上四个子节点，实际上最多增加3个Node
        {
            // 即将满足关键点的数量要求，采用更优化的方法计算，先进行排序
            while(!bFinish)
            {

                prevSize = lNodes.size();

                vector<pair<int,ExtractorNode*> > vPrevSizeAndPointerToNode = vSizeAndPointerToNode;
                vSizeAndPointerToNode.clear();

                // 按照int数据的大小，对pair进行升序排序
                sort(vPrevSizeAndPointerToNode.begin(),vPrevSizeAndPointerToNode.end());
                for(int j=vPrevSizeAndPointerToNode.size()-1;j>=0;j--)
                {
                    // 从后往前检查是否可以分裂，因为后面的元素size值较大，更有可能可以分裂
                    ExtractorNode n1,n2,n3,n4;
                    vPrevSizeAndPointerToNode[j].second->DivideNode(n1,n2,n3,n4);

                    // Add childs if they contain points
                    if(n1.vKeys.size()>0)
                    {
                        lNodes.push_front(n1);
                        if(n1.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n1.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n2.vKeys.size()>0)
                    {
                        lNodes.push_front(n2);
                        if(n2.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n2.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n3.vKeys.size()>0)
                    {
                        lNodes.push_front(n3);
                        if(n3.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n3.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }
                    if(n4.vKeys.size()>0)
                    {
                        lNodes.push_front(n4);
                        if(n4.vKeys.size()>1)
                        {
                            vSizeAndPointerToNode.push_back(make_pair(n4.vKeys.size(),&lNodes.front()));
                            lNodes.front().lit = lNodes.begin();
                        }
                    }

                    // 每个Node的成员变量lit，是一个iterator，记录了自己在列表里的位置，在这里用到
                    lNodes.erase(vPrevSizeAndPointerToNode[j].second->lit);

                    if((int)lNodes.size()>=N)
                        break;
                }

                if((int)lNodes.size()>=N || (int)lNodes.size()==prevSize)
                    bFinish = true;

            }
        }
    }

    // Retain the best point in each node
    // 从每个Node中选出最优的关键点
    vector<cv::KeyPoint> vResultKeys;
    vResultKeys.reserve(nfeatures);
    for(list<ExtractorNode>::iterator lit=lNodes.begin(); lit!=lNodes.end(); lit++)
    {
        vector<cv::KeyPoint> &vNodeKeys = lit->vKeys;
        cv::KeyPoint* pKP = &vNodeKeys[0];
        float maxResponse = pKP->response;
        // cv::KeyPoint类有一个成员变量response来表示关键点的强度
        // 遍历，找到其中response最大的关键点，加入到结果当中
        for(size_t k=1;k<vNodeKeys.size();k++)
        {
            if(vNodeKeys[k].response>maxResponse)
            {
                pKP = &vNodeKeys[k];
                maxResponse = vNodeKeys[k].response;
            }
        }

        vResultKeys.push_back(*pKP);
    }

    return vResultKeys;
}

void ORBextractor::ComputeKeyPointsOctTree(vector<vector<KeyPoint> >& allKeypoints)
{
    // allKeypoints是一个二维的数组，第一维的维度是金字塔的层数
    allKeypoints.resize(nlevels);
    // 初步设定每个Cell内的像素数量 W * W 个
    const float W = 30;

    for (int level = 0; level < nlevels; ++level)
    {
        // 为了提取FAST特征点，在BORDER中取出3个像素单位
        // 图片相比于mvImagePyramid[level]四周向外扩了3个像素
        const int minBorderX = EDGE_THRESHOLD-3;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD+3;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD+3;

        // 存储需要平均分配的特征点
        vector<cv::KeyPoint> vToDistributeKeys;
        // 一般都是过量采集，所以预分配空间是10倍
        vToDistributeKeys.reserve(nfeatures*10);

        const float width = (maxBorderX-minBorderX);
        const float height = (maxBorderY-minBorderY);

        const int nCols = width/W;
        const int nRows = height/W;
        // 每个Cell内的像素数量是 hCell * wCell
        // wCell和hCell计算出来可能会小于30 
        const int wCell = ceil(width/nCols);
        const int hCell = ceil(height/nRows);

        for(int i=0; i<nRows; i++)
        {
            const float iniY =minBorderY+i*hCell;
            // (+6)是因为FAST特征点计算时，取的是半径为3的圆
            float maxY = iniY+hCell+6;

            // 如果iniY超出下边界，则跳出循环，这里 continue == break
            if(iniY>=maxBorderY-3)
                continue;
            // 如果计算出的下边界超出实际最大可能的下边界，则以实际边界为准
            if(maxY>maxBorderY)
                maxY = maxBorderY;

            for(int j=0; j<nCols; j++)
            {
                const float iniX =minBorderX+j*wCell;
                float maxX = iniX+wCell+6;
                // 如果iniX超出右边界，则跳出循环，这里 continue == break
                if(iniX>=maxBorderX-3)
                    continue;
                // 如果计算出的右边界超出实际最大可能的右边界，则以实际边界为准
                if(maxX>maxBorderX)
                    maxX = maxBorderX;

                vector<cv::KeyPoint> vKeysCell;
                // ! 提取FAST角点的第一个参数有误
                FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX), // 输入的图像
                     vKeysCell, // 存储关键点的向量
                     iniThFAST, // 初始使用的FAST阈值-20，
                     true);     // 采用非极大值抑制，避免出现两个关键点紧邻的情况

                // 如果没有检测到关键点，则降低阈值
                if(vKeysCell.empty())
                {
                    FAST(mvImagePyramid[level].rowRange(iniY,maxY).colRange(iniX,maxX),
                         vKeysCell,minThFAST,true);
                }

                // 当图像检测到FAST角点时，执行以下代码
                if(!vKeysCell.empty())
                {
                    // 遍历所有检测到的FAST角点
                    for(vector<cv::KeyPoint>::iterator vit=vKeysCell.begin(); vit!=vKeysCell.end();vit++)
                    {
                        // 这些角点的坐标是基于图像Cell的，要把它们恢复到之前的坐标系下
                        // 下面的八叉树整理特征点的时候会用到
                        (*vit).pt.x+=j*wCell;
                        (*vit).pt.y+=i*hCell;
                        // 添加到待整理的列表中
                        vToDistributeKeys.push_back(*vit);
                    }
                }

            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nfeatures);

        // 使用四叉树法对特征点进行平均分配
        keypoints = DistributeOctTree(vToDistributeKeys, minBorderX, maxBorderX,
                                      minBorderY, maxBorderY,mnFeaturesPerLevel[level], level);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Add border to coordinates and scale information
        const int nkps = keypoints.size();
        for(int i=0; i<nkps ; i++)
        {
            keypoints[i].pt.x+=minBorderX;
            keypoints[i].pt.y+=minBorderY;
            keypoints[i].octave=level;
            keypoints[i].size = scaledPatchSize;
        }
    }

    // compute orientations
    // 计算关键点方向，给每个KP的angle变量赋值
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

void ORBextractor::ComputeKeyPointsOld(std::vector<std::vector<KeyPoint> > &allKeypoints)
{
    allKeypoints.resize(nlevels);

    float imageRatio = (float)mvImagePyramid[0].cols/mvImagePyramid[0].rows;

    for (int level = 0; level < nlevels; ++level)
    {
        const int nDesiredFeatures = mnFeaturesPerLevel[level];

        const int levelCols = sqrt((float)nDesiredFeatures/(5*imageRatio));
        const int levelRows = imageRatio*levelCols;

        const int minBorderX = EDGE_THRESHOLD;
        const int minBorderY = minBorderX;
        const int maxBorderX = mvImagePyramid[level].cols-EDGE_THRESHOLD;
        const int maxBorderY = mvImagePyramid[level].rows-EDGE_THRESHOLD;

        const int W = maxBorderX - minBorderX;
        const int H = maxBorderY - minBorderY;
        const int cellW = ceil((float)W/levelCols);
        const int cellH = ceil((float)H/levelRows);

        const int nCells = levelRows*levelCols;
        const int nfeaturesCell = ceil((float)nDesiredFeatures/nCells);

        vector<vector<vector<KeyPoint> > > cellKeyPoints(levelRows, vector<vector<KeyPoint> >(levelCols));

        vector<vector<int> > nToRetain(levelRows,vector<int>(levelCols,0));
        vector<vector<int> > nTotal(levelRows,vector<int>(levelCols,0));
        vector<vector<bool> > bNoMore(levelRows,vector<bool>(levelCols,false));
        vector<int> iniXCol(levelCols);
        vector<int> iniYRow(levelRows);
        int nNoMore = 0;
        int nToDistribute = 0;


        float hY = cellH + 6;

        for(int i=0; i<levelRows; i++)
        {
            const float iniY = minBorderY + i*cellH - 3;
            iniYRow[i] = iniY;

            if(i == levelRows-1)
            {
                hY = maxBorderY+3-iniY;
                if(hY<=0)
                    continue;
            }

            float hX = cellW + 6;

            for(int j=0; j<levelCols; j++)
            {
                float iniX;

                if(i==0)
                {
                    iniX = minBorderX + j*cellW - 3;
                    iniXCol[j] = iniX;
                }
                else
                {
                    iniX = iniXCol[j];
                }


                if(j == levelCols-1)
                {
                    hX = maxBorderX+3-iniX;
                    if(hX<=0)
                        continue;
                }


                Mat cellImage = mvImagePyramid[level].rowRange(iniY,iniY+hY).colRange(iniX,iniX+hX);

                cellKeyPoints[i][j].reserve(nfeaturesCell*5);

                FAST(cellImage,cellKeyPoints[i][j],iniThFAST,true);

                if(cellKeyPoints[i][j].size()<=3)
                {
                    cellKeyPoints[i][j].clear();

                    FAST(cellImage,cellKeyPoints[i][j],minThFAST,true);
                }


                const int nKeys = cellKeyPoints[i][j].size();
                nTotal[i][j] = nKeys;

                if(nKeys>nfeaturesCell)
                {
                    nToRetain[i][j] = nfeaturesCell;
                    bNoMore[i][j] = false;
                }
                else
                {
                    nToRetain[i][j] = nKeys;
                    nToDistribute += nfeaturesCell-nKeys;
                    bNoMore[i][j] = true;
                    nNoMore++;
                }

            }
        }


        // Retain by score

        while(nToDistribute>0 && nNoMore<nCells)
        {
            int nNewFeaturesCell = nfeaturesCell + ceil((float)nToDistribute/(nCells-nNoMore));
            nToDistribute = 0;

            for(int i=0; i<levelRows; i++)
            {
                for(int j=0; j<levelCols; j++)
                {
                    if(!bNoMore[i][j])
                    {
                        if(nTotal[i][j]>nNewFeaturesCell)
                        {
                            nToRetain[i][j] = nNewFeaturesCell;
                            bNoMore[i][j] = false;
                        }
                        else
                        {
                            nToRetain[i][j] = nTotal[i][j];
                            nToDistribute += nNewFeaturesCell-nTotal[i][j];
                            bNoMore[i][j] = true;
                            nNoMore++;
                        }
                    }
                }
            }
        }

        vector<KeyPoint> & keypoints = allKeypoints[level];
        keypoints.reserve(nDesiredFeatures*2);

        const int scaledPatchSize = PATCH_SIZE*mvScaleFactor[level];

        // Retain by score and transform coordinates
        for(int i=0; i<levelRows; i++)
        {
            for(int j=0; j<levelCols; j++)
            {
                vector<KeyPoint> &keysCell = cellKeyPoints[i][j];
                KeyPointsFilter::retainBest(keysCell,nToRetain[i][j]);
                if((int)keysCell.size()>nToRetain[i][j])
                    keysCell.resize(nToRetain[i][j]);


                for(size_t k=0, kend=keysCell.size(); k<kend; k++)
                {
                    keysCell[k].pt.x+=iniXCol[j];
                    keysCell[k].pt.y+=iniYRow[i];
                    keysCell[k].octave=level;
                    keysCell[k].size = scaledPatchSize;
                    keypoints.push_back(keysCell[k]);
                }
            }
        }

        if((int)keypoints.size()>nDesiredFeatures)
        {
            KeyPointsFilter::retainBest(keypoints,nDesiredFeatures);
            keypoints.resize(nDesiredFeatures);
        }
    }

    // and compute orientations
    for (int level = 0; level < nlevels; ++level)
        computeOrientation(mvImagePyramid[level], allKeypoints[level], umax);
}

/**
 * @brief 计算图像某层金字塔上特征点的描述子
 * 
 * @param[in] image             某层金字塔图像
 * @param[in] keypoints         特征点vector容器
 * @param[out] descriptors      描述子
 * @param[in] pattern           计算描述子所使用的固定随机点数
 **/
static void computeDescriptors(const Mat& image, vector<KeyPoint>& keypoints, Mat& descriptors,
                               const vector<Point>& pattern)
{
    // 创建一个 行数为关键点数量，列数为32，类型为unsigned char的矩阵
    descriptors = Mat::zeros((int)keypoints.size(), 32, CV_8UC1);

    for (size_t i = 0; i < keypoints.size(); i++)
        computeOrbDescriptor(keypoints[i], image, &pattern[0], descriptors.ptr((int)i));
}

/**
 * @brief 提取ORB特征点和描述子的仿函数，通过重载()实现
 * 
 * @param[in] _image                输入原始图的图像
 * @param[in] _mask                 掩膜mask
 * @param[in&out] _keypoints      存储特征点关键点的向量
 * @param[in&out] _descriptors    存储特征点描述子的向量
 **/
void ORBextractor::operator()( 
    InputArray _image, 
    InputArray _mask, 
    vector<KeyPoint>& _keypoints,
    OutputArray _descriptors)
{ 
    if(_image.empty())
        return;

    Mat image = _image.getMat();
    assert(image.type() == CV_8UC1 );

    // Pre-compute the scale pyramid
    // 预处理，图像金字塔每一层进行了缩放，构建图像金字塔
    ComputePyramid(image);
    
    //==============特征点的提取和分配================
    vector < vector<KeyPoint> > allKeypoints;
    ComputeKeyPointsOctTree(allKeypoints);
    //ComputeKeyPointsOld(allKeypoints);

    //=======拷贝图像描述子到新的矩阵descriptors=======
    Mat descriptors;

    // 统计整个图像金字塔的关键点数量
    int nkeypoints = 0;
    for (int level = 0; level < nlevels; ++level)
        nkeypoints += (int)allKeypoints[level].size();

    // 如果本图像金字塔没有任何的特征点，则强制释放矩阵的数据
    if( nkeypoints == 0 )
        _descriptors.release();
    else
    {
        // 如果有特征点，那么创建存储描述子的矩阵，存储的数量是整个图像金字塔的
        _descriptors.create(nkeypoints,     // 行数
                            32,             // 列数，32*8=256位描述子
                            CV_8U);         // 类型
        // ? 不是在原矩阵上修改，而是复制了一个新的矩阵
        descriptors = _descriptors.getMat();
    }

    // 清空了原传入的矩阵
    _keypoints.clear();
    // 并预分配正确的大小空间
    _keypoints.reserve(nkeypoints);

    // descriptors存储的是整个金字塔的描述子，所有使用一个offset记录每一层关键点的位置便宜
    // 辅助进行遍历
    int offset = 0;
    // 遍历图像金字塔的每一层
    for (int level = 0; level < nlevels; ++level)
    {
        // 获取在allKeypoints中当前层的特征点的容器的句柄
        vector<KeyPoint>& keypoints = allKeypoints[level];
        // 本层的特征掉数量
        int nkeypointsLevel = (int)keypoints.size();

        // 如果特征点数量为0，则跳出本层循环，进行下一层的检测
        if(nkeypointsLevel==0)
            continue;

        // preprocess the resized image  
        //=============对图像进行高斯模糊===============
        // 获取图像金字塔每一层图像的深拷贝
        Mat workingMat = mvImagePyramid[level].clone();
        // Note:提取特征点的时候，用的是清晰的原图像，计算描述子，为了避免图像噪声的影响，进行了高斯模糊
        GaussianBlur(workingMat,            // 源图像
                    workingMat,             // 输出图像
                    Size(7, 7),             // 高斯滤波器Kernel的大小，必须为正奇数
                    2,                      // x方向的标准差
                    2,                      // y方向的标准差
                    BORDER_REFLECT_101);    // 边缘拓展点的插值类型

        // Compute the descriptors  计算描述子
        // desc存储当前层的描述子
        Mat desc = descriptors.rowRange(offset, offset + nkeypointsLevel);
        // ============计算高斯模糊后的描述子===========
        computeDescriptors(workingMat,      // 高斯模糊后的图像
                            keypoints,      // 当前图层特征点集合
                            desc,           // 存储计算之后的描述子
                            pattern);       // 随机采样点集

        // 更新偏移量的值，加上当前层所有特征点的数目
        offset += nkeypointsLevel;

        // Scale keypoint coordinates
        // ======对所有非第0层的图像的坐标恢复到第0层=====
        if (level != 0)
        {
            // 获取当前层的缩放系数，1.2^n
            float scale = mvScaleFactor[level]; //getScale(level, firstLevel, scaleFactor);
            for (vector<KeyPoint>::iterator keypoint = keypoints.begin(),
                 keypointEnd = keypoints.end(); keypoint != keypointEnd; ++keypoint)
                // 特征点本身直接乘上缩放倍数,重载等价于分别对pt.x和pt.y乘上缩放系数
                keypoint->pt *= scale;
        }
        // And add the keypoints to the output
        // 把keypoints中内容插入到_keypoints的末尾
        // keypoints是对allKeypoints每个点的引用，这里直接转存到_keypoints中
        _keypoints.insert(_keypoints.end(), keypoints.begin(), keypoints.end());
    }
}

void ORBextractor::ComputePyramid(cv::Mat image)
{
    for (int level = 0; level < nlevels; ++level)
    {
        // Scale = 1.2 , InvScale = 1/1.2
        // 最底部是第0层，最上面是第7层，最上面的面积小
        float scale = mvInvScaleFactor[level];

        // 根据每一层的缩放系数，计算图像的大小
        // 第一个参数是 Width，第二个参数的是 Height
        Size sz(cvRound((float)image.cols*scale), cvRound((float)image.rows*scale));

        // 图像两边分别预留一个边界，计算图像尺寸
        Size wholeSize(sz.width + EDGE_THRESHOLD*2, sz.height + EDGE_THRESHOLD*2);
        
        // 后者的掩膜没有被用到
        Mat temp(wholeSize, image.type()), masktemp;
        
        // 从 temp 图像中截取ROI区域，4个系数分别是左上角的x坐标，y坐标，宽度和高度
        // 从上一层拷贝过来的图像应该存放在ROI区域
        // 这是一个浅拷贝，拷贝给了成员变量，函数结束之后不会释放
        mvImagePyramid[level] = temp(Rect(EDGE_THRESHOLD, EDGE_THRESHOLD, sz.width, sz.height));

        // Compute the resized image
        if( level != 0 )
        {
            // 对于上层图像
            // Step 1. 根据上一层图像的像素值，把图像缩放到期望的大小
            resize(
                mvImagePyramid[level-1],    // 输入的图像矩阵
                mvImagePyramid[level],      // 输出的图像矩阵
                sz,                         // 输出图像的大小，需要与上面矩阵的大小匹配
                0,                          // x方向的缩放系数，默认为0，表示不起作用，由第三个参数决定大小
                0,                          // y方向的缩放系数，同上
                INTER_LINEAR);              // 双线性插值方式

            // Step 2. 给当前层的图像加上BORDER
            copyMakeBorder(
                mvImagePyramid[level],      // 输入的图像；该函数把输入图像放到输出图像的正中间 ...
                temp,                       // 输出的图像；如果已经在正中间，就只是添加BORDER
                EDGE_THRESHOLD,             
                EDGE_THRESHOLD, 
                EDGE_THRESHOLD, 
                EDGE_THRESHOLD,             // 以上四个分别为上、下、左、右边界值
                BORDER_REFLECT_101+BORDER_ISOLATED);    // BORDER的类型
        }
        else
        {
            // 第一层图像金字塔不需要缩放，只需要给Temp矩阵加上边界BORDER
            // ! BORDER的作用还没有体现出来 EDGE_THRESHOLD
            copyMakeBorder(image, temp, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD, EDGE_THRESHOLD,
                           BORDER_REFLECT_101);            
        }
    }

}

} //namespace ORB_SLAM
