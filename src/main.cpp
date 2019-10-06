#include <iostream>
#include "opencv2/opencv.hpp"
#include <math.h>
#include <sstream>
#include "network.h"
using namespace std;
using namespace cv;
#define widthThreshold 45
#define heigtThreshold 26

string int_to_str(int i)
{
    stringstream s;
    s<<i;
    return s.str();
}
int myDiscern(Mat n)
{
	//1的图像，使用穿线会是8。应该从它的尺寸入手，高远大于宽，这里我们选取3倍比.
//    if(3*n.cols<n.rows)
  //  {
  //      cout<<'1';
   //     return 1;
   // }
    //竖线
    int x_half=n.cols/2;
    //上横线
    int y_one_third=n.rows/3;
    //下横线
    int y_two_third=n.rows*2/3;
    //每段数码管，0灭，1亮
    int a=0,b=0,c=0,d=0,e=0,f=0,g=0;
	
	//竖线识别a,g,d段
    for(int i=0;i<n.rows;i++)
    {
        uchar *data=n.ptr<uchar>(i);
        if(i<y_one_third)
        {
            if(data[x_half]==255) a=1;
        }
        else if(i>y_one_third&&i<y_two_third)
        {
            if(data[x_half]==255) g=1;
        }
        else
        {
            if(data[x_half]==255) d=1;
        }
    }

	//上横线识别：
    for(int j=0;j<n.cols;j++)
    {
        uchar *data=n.ptr<uchar>(y_one_third);
        //f
        if(j<x_half)
        {
            if(data[j]==255) f=1;
        }
        //b
        else
        {
            if(data[j]==255) b=1;
        }
    }
	
	//下横线识别：
    for(int j=0;j<n.cols;j++)
    {
        uchar *data=n.ptr<uchar>(y_two_third);
		//e
        if(j<x_half)
        {
            if(data[j]==255) e=1;
        }
        //c
        else
        {
            if(data[j]==255) c=1;
        }
    }
	
	//七段管组成的数字
    if(a==1 && b==1 && c==1 && d==1 && e==1 && f==1 && g==0)
    {
       // cout<<"0";
        return 0    ;
    }
    /*
    else if(a==0 && b==1 && c==1 && d==0 && e==0 && f==0 && g==0)
    {
        cout<<"1";
        return 1;
    }
    */
    else if(a==1 && b==1 && c==0 && d==1 && e==1 && f==0 && g==1)
    {
       // cout<<"2";
        return 2;
    }
    else if(a==1 && b==1 && c==1 && d==1 && e==0 && f==0 && g==1)
    {
       // cout<<"3";
        return 3;
    }
    else if(a==0 && b==1 && c==1 && d==0 && e==0 && f==1 && g==1)
    {
       // cout<<"4";
        return 4;
    }
    else if(a==1 && b==0 && c==1 && d==1 && e==0 && f==1 && g==1)
    {
      //  cout<<"5";
        return 5;
    }
    else if(a==1 && b==0 && c==1 && d==1 && e==1 && f==1 && g==1)
    {
      //  cout<<"6";
        return 6;
    }
    else if(a==1 && b==1 && c==1 && d==0 && e==0 && f==0 && g==0)
    {
     //   cout<<"7";
        return 7;
    }
    else if(a==1 && b==1 && c==1 && d==1 && e==1 && f==1 && g==1)
    {
     //   cout<<"8";
        return 8;
    }
    else if(a==1 && b==1 && c==1 && d==1 && e==0 && f==1 && g==1)
    {
    //   /  cout<<"9";
        return 9;
    }
    else
    {
    //   /  printf("[error_%d_%d_%d_%d_%d_%d_%d]",a,b,c,d,e,f,g);
    }

}
//九宫格中的一个小格
struct SudokuGrid
{
    Mat grid;
    Point2f gridCenter;
    long evaluation = 0;
    int lable;//聚类标签
    Point2f originSize;
};

int main()
{
    int num_jpg=0;
    int networkInputs=28*28; //网络参数设置
    int networkOutputs=10;
    int epoches=10;
    float learningRate=0.1;

    Network *network = new Network(epoches,learningRate,networkInputs,networkOutputs);
    network->addLayer(256,SIGMOID); //加入全连接层，参数有神经元个数和激活函数类型  256
    network->addLayer(128,SIGMOID);//128
    network->addLayer(network->mNumOutputs,SIGMOID);
    ifstream infile("mnist.weight");
    if(!infile.is_open()){
        cout<<"open weight file failed!"<<endl;
        exit(-1);
    }
    for(int i=0;i<network->mNumLayers;i++){
        Layer *layer=network->mLayers[i];
        for(int m=0;m<layer->mNumNodes;m++){
            for(int n=0;n<layer->mNumInputNodes+1;n++){
                infile>>layer->mWeights[m][n];
            }
        }
    }
    infile.close();
    cout<<"load weight from <"<<"minst.weight"<<"> done."<<endl;
    //loadWeight("mnist.weight",network);
    network->mTrain=false;
            
        
    VideoCapture cap("buff1.avi");
    Mat src,test,gray,thresh,thresh2,original,src_shuma,shuma,cx_src;
    vector<Mat>channels;
    int a = 170;//canny参数170
    int b= 11;
    Mat element1 = getStructuringElement(MORPH_RECT, Size(3, 3));   
    Mat element2 = getStructuringElement(MORPH_RECT, Size(7, 7));
    for(;;)
    {   

        cap.read(src);
        original=src.clone();
        shuma=src.clone();
        Rect rect(100, 113, 450, 350);
        Rect rect_shuma(0, 0, 500, 113);
        src = src(rect);
        src_shuma=shuma(rect_shuma);
        split(src_shuma,channels);
        Mat Red =channels.at(2);
        Mat Blue=channels.at(0);
        for(int i=0;i<Red.rows;i++)
        {
            for(int j=0;j<Red.cols;j++)
            {
                if(Red.at<uchar>(i,j)>150)//&&(Red.at<uchar>(i,j)>1.1*Blue.at<uchar>(i,j)))
                {
                    Red.at<uchar>(i,j)=255;
                }
                else Red.at<uchar>(i,j)=0;
            }
        }
        vector<vector<Point> >contours;
        threshold(Red,cx_src,150,255,THRESH_BINARY);
        dilate(cx_src, cx_src, element2, Point(-1, -1));
        findContours(cx_src,contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
       
        //src.copyTo(gray);
        //GaussianBlur(src,src,Size(3,3),0,0);
        //blur(src,src,Size(1,1),Point(-1,-1));
        cvtColor(src, gray, CV_BGR2GRAY);
	    
        threshold(gray,thresh, 85, 255, THRESH_BINARY_INV);
        thresh=~thresh;
        //Canny(gray, thresh, a, 3* a, 3);
        vector<vector<Point> >contours0;
        vector<RotatedRect> rectBorder;

        vector<SudokuGrid> sudoku;//九宫格
        
    //    erode(thresh,thresh,element1,Point(-1,-1));
       // dilate(thresh, thresh, element2, Point(-1, -1));
        findContours(thresh, contours0, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
        //test=thresh.clone();
        drawContours(thresh, contours0, -1, Scalar(255, 255, 255));
        
        Rect rect2;
        Rect rect3;
       // Rect rect4;
        if (contours0.size() > 0)
        {       

                for (size_t j = 0; j < contours.size(); j++)
                {
                    rect3= boundingRect(Mat(contours[j]));
                    //rect4=Rect(rect3.x-20,rect3.y-20,rect3.width+20,rect3.height+20);
                    if(rect3.height<20||rect3.width<20)
                        continue;
                    rectangle(original, rect3, Scalar(0, 0, 255), 3);
                    Mat roi_shuma(cx_src, rect3); 
                    int result_shumaguan ;
                    result_shumaguan = myDiscern(roi_shuma);
                    putText(original,int_to_str(result_shumaguan),Point(rect3.x+10,rect3.y),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255),4,8);
/*  穿线法识别数码管交点法*/

                  /*  string symbol=".jpg";
			          string path="/home/chunran/mnist-master/testData/";
			            if(num_jpg<10)
			        {
                    imwrite(int_to_str(num_jpg)+symbol,roi_shuma);
			        }
                  */// num_jpg++;
                  /*
                    Mat row1,row2,col1;
	                row1 = roi_shuma.rowRange(roi_shuma.rows/3,roi_shuma.rows/3+1);
	                row2 = roi_shuma.rowRange(2*roi_shuma.rows/3,2*roi_shuma.rows/3+1);
	                col1 = roi_shuma.colRange(roi_shuma.cols/2,roi_shuma.cols/2+1);
                    imshow("roi",roi_shuma);
	                //cout<<row1<<endl<<endl;
	                int flag_row1=0;
	                int flag_row2=0;
	                int flag_col1=0;
	                int point_row1[10],point_row2[10],point_col1[10];
	            for(int i=0;i<row1.cols-1;i++)
	    {
		    if(abs(row1.at<uchar>(0,i)-row1.at<uchar>(0,i+1))==255)
		    {
			point_row1[flag_row1]=i;
			flag_row1++;
		    }
			if(abs(row2.at<uchar>(0,i)-row2.at<uchar>(0,i+1))==255)
		    {
			point_row2[flag_row2]=i;
			flag_row2++;
		    }
	    }
	            for(int j=0;j<col1.rows-1;j++)
	    {
		if(abs(col1.at<uchar>(j,0)-col1.at<uchar>(j+1,0))==255)
		{
			point_col1[flag_col1]=j;
			flag_col1++;
		}
	}   
	cout<<flag_row1<<endl;
	cout<<flag_row2<<endl;
	cout<<flag_col1<<endl;
        
	    if(flag_row1==2&&flag_row2==2&&flag_col1==2)
	    {
		result_shumaguan=1;
        putText(original,int_to_str(result_shumaguan),Point(rect3.x,rect3.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
        
	    }
	    if(flag_row1==2&&flag_row2==2&&flag_col1==6)
	    {
		if(point_row1[0]>row1.cols/2)
		{
			if(point_row2[0]>row2.cols/2)
			{
				result_shumaguan=3;
                putText(original,int_to_str(result_shumaguan),Point(rect3.x,rect3.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
			}
			else 
            {
                
            
				result_shumaguan=2;
                putText(original,int_to_str(result_shumaguan),Point(rect3.x,rect3.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
            }
		}
		else
        {

        
			result_shumaguan=5;
            putText(original,int_to_str(result_shumaguan),Point(rect3.x,rect3.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
        }
		
	    }
	    if(flag_row1==4&&flag_row2==4&&flag_col1==6)
	    {
		result_shumaguan=8;
        putText(original,int_to_str(result_shumaguan),Point(rect3.x,rect3.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
	    }
	    if(flag_row1==2&&flag_row2==4&&flag_col1==6)
	    {
		result_shumaguan=6;
        putText(original,int_to_str(result_shumaguan),Point(rect3.x,rect3.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
	    }   
	    if(flag_row1==4&&flag_row2==2&&flag_col1==6)
	    {
		result_shumaguan=9;
        putText(original,int_to_str(result_shumaguan),Point(rect3.x,rect3.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
	    }
	    //if(flag_row1==2&&flag_row2==2&&point_row2[1]>row2.cols/2&&point_row1[1]>row1.cols/2&&flag_col1<6)
	  //  {
	//			result_shumaguan=7;
       //         putText(original,int_to_str(result_shumaguan),Point(rect3.x,rect3.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
	  //  }
        
	    //cout<<result_shumaguan<<endl;
        */

                }
                vector<Point> poly;
                for (size_t t = 0; t < contours0.size(); t++)
                {
                    
                    approxPolyDP(contours0[t], poly, 5, true);
                   // if (poly.size() == 4)
                    //{                  
                    rect2= boundingRect(Mat(contours0[t]));
                    rectBorder.push_back(minAreaRect(Mat(contours0[t])));

                    if(rect2.width<20||rect2.width>200||rect2.height>200||rect2.height<50)
                    {
                        continue;
                    }
                    
                    else
                    {  
                        SudokuGrid temp;
                        Rect rect_change=Rect((rect2.x+100),(rect2.y+113),rect2.width,rect2.height);
                       // rectangle(src, rect2, Scalar(0, 255, 0), 3);

                        rectangle(original, rect_change, Scalar(0, 255, 0), 3);
                        temp.originSize = Point2f(rect2.width, rect2.height);
                        Mat roi(thresh, rect2);
                        resize(roi, roi, Size(28, 28));//压缩
                        temp.grid = roi;

                     //   string symbol=".jpg";
			          //  string path="/home/chunran/mnist-master/testData/";
			        //    if(num_jpg<100)
			       // {
                       // imwrite(int_to_str(num_jpg)+symbol,roi);
			       // }
                        temp.gridCenter = Point2f(rect2.x, rect2.y);
                        sudoku.push_back(temp);
			            num_jpg++;

                        float *d=new float[28*28];
                        for(int i=0;i<28;i++)
                        {
                        for(int j=0;j<28;j++)
                        {
                            float x=(roi.at<uchar>(i,j))/255.0; //将二维像素值转成一维向量，并归一化
                            d[i*28+j]=x;
                        }
                        }
                            
                        float max=-9999;
                        int idx=10;
                        network->compute(d); //开始预测
                        float *out=network->mOutputs; //获得网络输出
                        for(int i=1;i<10;i++)
                        {
                        cout<<out[i]<<",";
                        if(out[i]>max)
                        { //取最大输出为预测值
                           max=out[i];
                           idx=i;
                         
                        }
                        }
                        cout<<"the prediction is: "<<idx<<endl;
                        cout<<endl;
                        string result=int_to_str(idx);
                      //  putText(src,result,Point(rect2.x,rect2.y),FONT_HERSHEY_SIMPLEX,2,Scalar(0,0,255),4,8);
                        putText(original,result,Point(rect2.x+110,rect2.y+113),FONT_HERSHEY_SIMPLEX,1,Scalar(255,255,255),4,8);
                    }
            

                }
                    
                    //   }
                    
        }
       
        
        /*
         if (sudoku.size() > 9)//检测到大于9个，进行剔除
                {

                   vector<Point2f> rectSize;//对矩形尺寸聚类

                    for (size_t t = 0; t < sudoku.size(); t++)
                        rectSize.push_back(sudoku[t].originSize);

                    const int K = 2;//聚类的类别数量，即Lable的取值
                    Mat Label, Center;
                    kmeans(rectSize, K, Label, TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 10, 1.0),
                    1, KMEANS_PP_CENTERS, Center);//聚类3次，取结果最好的那次，聚类的初始化采用PP特定的随机算法。

                   int LableNum[K] = { 0 };//不同类别包含的成员数量，如类别0有8个，类别1有2个。。。。
                   int sudoLable;//代表九宫格的标签
                   for (int i = 0; i < rectSize.size(); ++i)
                   {
                       sudoku[i].lable = Label.at<int>(i);

                       cout << rectSize[i] << "\t-->\t" << sudoku[i].lable << endl;

                       for (int j = 0; j < K; ++j)
                           if (sudoku[i].lable == j)
                           {
                               LableNum[j]++;
                               break;
                           }
                   }
                   sudoLable = *max_element(LableNum, LableNum + K);

                   for (int j = 0; j < K; ++j)
                       if (LableNum[j] == sudoLable)
                       {
                           sudoLable = j;
                           break;
                       }
                    vector<SudokuGrid> sudokuTemp;//九宫格
                    for (int i = 0; i < rectSize.size(); ++i)
                       if (sudoku[i].lable == sudoLable)
                           sudokuTemp.push_back(sudoku[i]);
                    sudoku.swap(sudokuTemp);
                }
        for (size_t t = 0; t < sudoku.size(); t++)
        {
            rectangle(src,Rect(sudoku[t].gridCenter.x,sudoku[t].gridCenter.y,sudoku[t].originSize.x,sudoku[t].originSize.y),Scalar(0, 255, 0), 3);
        }
        */
      //  namedWindow("src",CV_WINDOW_AUTOSIZE);
      //  namedWindow("thresh",CV_WINDOW_AUTOSIZE);
        imshow("src",src);
        imshow("thresh", thresh);
        imshow("original",original);
        imshow("src_shuma",src_shuma);
        imshow("cx_src",cx_src);
        
        char c=waitKey(100);
        if(c==27)
        {
            break;
        }
    }
return 0;
}
            
