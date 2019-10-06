//network.cpp
#include "network.h"
#include <cstring>
#include <stdlib.h>
#include <math.h>
#include <iostream>
using namespace std;


Network::Network(int epoches,float learningRate, int numInputs, int numOutputs)
    :mEpoches(epoches),
    mNumInputs(numInputs),
    mNumOutputs(numOutputs),
    mLearningRate(learningRate)
{
    mNumLayers=0;
    mErrorSum=0;
    mInputs=NULL;
    mOutputs=NULL;
}

Network::~Network()
{
    for (int i = 0; i < mNumLayers; i++)
    {
        if (mLayers[i])
        {
            delete mLayers[i];
        }
    }
}

void Network::init() //初始化
{
    for (int i = 0; i < mNumLayers; ++i)
    {
        mLayers[i]->init();
    }
    mErrorSum = 0;
}

void Network::addLayer(int numNodes,ACTIVATION activate) //添加全连接层
{
    int numInputNodes = (mNumLayers > 0) ? mLayers[mNumLayers-1]->mNumNodes : mNumInputs;
    mLayers.push_back(new Layer(numNodes,numInputNodes,activate));
    mNumLayers++;
}

void Network::forwardNetwork(float *inputs,int label) //网络前向计算
{
    for (int i = 0; i < mNumLayers; i++)
    {
        mLayers[i]->forwardLayer(inputs); //对每个层计算
        inputs = mLayers[i]->mOutputs;
    }
    mOutputs=inputs; //注意是指向了最后一层的输出
    if(!mTrain) return;

    float *outputs = mOutputs;
    float *delta = mLayers[mNumLayers-1]->mDelta;
    for (int i = 0; i < mNumOutputs; i++)
    {
        float err;
        if(i==label){
            err=1-outputs[i];
        }else{
            err=0-outputs[i];
        }
        delta[i] = err; //计算 delta 和误差
        mErrorSum += err * err;
    }
}

void Network::backwardNetwork() //网络反向计算并更新权重参数
{
    float *prevOutputs = NULL;
    float *prevDelta = NULL;

    for (int i = mNumLayers-1; i >= 0; i--)
    {
        if (i > 0)
        {
            Layer &prev = *mLayers[i-1];
            prevOutputs = prev.mOutputs;
            prevDelta = prev.mDelta;
            memset(prevDelta, 0, prev.mNumNodes * sizeof(float));
        }
        else
        {
            prevOutputs = mInputs;
            prevDelta = NULL; //第一层前是输入，不需计算 delta
        }
        mLayers[i]->backwardLayer(prevOutputs, prevDelta,mLearningRate); //反向计算更新权重
    }
}

void Network::compute(float *inputs,int label) //网络计算入口
{
    mInputs=inputs;
    forwardNetwork(inputs,label);
    if(!mTrain){
        return;
    }
    backwardNetwork();
}
