//layer.cpp
#include<stdlib.h>
#include "layer.h"
#include <cstring>
#include <time.h>
#include <iostream>
#include <cmath>
using namespace std;

Layer::Layer(int numNodes, int numInputNodes,ACTIVATION activa)
	:mNumNodes(numNodes),
	mNumInputNodes(numInputNodes),
	mActivate(activa)
{
	mWeights = new float*[mNumNodes];
	mOutputs = new float[mNumNodes];
	mDelta = new float[mNumNodes];
	init();
}

Layer::Layer(Layer &layer)
	:mNumNodes(layer.mNumNodes),
	mNumInputNodes(layer.mNumInputNodes),
	mActivate(layer.mActivate)
{
	int size = mNumNodes * sizeof(float);
	memcpy(mOutputs, layer.mOutputs, size);
	memcpy(mDelta, layer.mDelta, size);
	for (int i = 0; i < mNumNodes; i++)
	{
		memcpy(mWeights[i], layer.mWeights[i], layer.mNumInputNodes+1);
	}
}

Layer::~Layer()
{
	for (int i = 0; i < mNumNodes; i++)
	{
		delete [] mWeights[i];
	}
	delete [] mWeights;
	delete [] mOutputs;
	delete [] mDelta;
}

void Layer::init()
{
	memset(mOutputs, 0, mNumNodes * sizeof(float));
	memset(mDelta, 0, mNumNodes * sizeof(float));
	srand(time(0));
	for (int i = 0; i < mNumNodes; ++i)
	{
		float *curWeights = new float[mNumInputNodes + 1];
		mWeights[i] = curWeights;
		for (int w = 0; w < mNumInputNodes + 1; w++) //还有一个 bias 值，所以加 1
		{
			curWeights[w] = rand() % 1000 * 0.001 - 0.5;
		}
	}
}

float Layer::active(float x,ACTIVATION activate) //激活函数
{
    switch(activate){
        case SIGMOID:
            return (1.0/(1.0+exp(-x)));
		case RELU:
			return x*(x>0);
		case LEAKY:
			return (x>0)?x:0.1*x;
        default:
            cout<<"no activation."<<endl;
            return x;
    }
}

float Layer::gradient(float x,ACTIVATION activate) //激活函数导数
{
    switch(activate){
        case SIGMOID:
            return x*(1.0-x);
		case RELU:
			return (x>0);
		case LEAKY:
			return (x>0)?1:0.1;
        default:
            cout<<"no activation."<<endl;
            return 1.0;
    }
}

void Layer::forwardLayer(float *inputs) //前向计算
{
	for (int n = 0; n < mNumNodes; ++n)
    {
        float *curWeights = mWeights[n];
        float x = 0;
        int k;

        for (k = 0; k < mNumInputNodes; ++k)
        {
            x += curWeights[k] * inputs[k];
        }
        x += curWeights[k];
        mOutputs[n] = active(x,mActivate);
    }
}

void Layer::backwardLayer(float *prevOutputs,float *prevDelta,float learningRate) //反向计算
{
    for (int i = 0; i < mNumNodes; i++)
    {
        float* curWeights = mWeights[i];
        float delta = mDelta[i] * gradient(mOutputs[i],mActivate);
        int w;
        for (w = 0; w < mNumInputNodes; w++)
        {
            if (prevDelta)
            {
                prevDelta[w] += curWeights[w] * delta;
            }
            curWeights[w] += delta * learningRate * prevOutputs[w]; //更新权重
        }
        curWeights[w] += delta * learningRate; //更新 bias
    }
}
