//network.h
#ifndef __NETWORK_H__
#define __NETWORK_H__

#include "layer.h"
#include <vector>
using namespace std;

class Network
{
public:
    Network(int epoches,float learningRate,int numInputs,int numOutputs);
    ~Network();
    void compute(float *inputs,int label=10);
    void addLayer(int numNodes,ACTIVATION activate=SIGMOID);
private:
    void init();
    void forwardNetwork(float *inputs,int label);
    void backwardNetwork();

public:
    bool mTrain;
    int mEpoches;
    int mNumInputs;
    int mNumOutputs;
    int mNumLayers;
    float mLearningRate;
    float mErrorSum;
    float *mInputs;
    float *mOutputs;
    vector<Layer *> mLayers;
};

#endif
