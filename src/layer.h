//layer.h
#ifndef __LAYER_H__
#define __LAYER_H__

typedef enum{
    SIGMOID,RELU,LEAKY
} ACTIVATION;

class Layer
{
public:
	Layer(int numNodes, int numInputNodes,ACTIVATION activate=SIGMOID);
	Layer(Layer &layer);
	~Layer();
	void forwardLayer(float *inputs);
	void backwardLayer(float *prevOutputs,float *prevDelta,float learningRate);
	void init();

private:
	inline float active(float x,ACTIVATION activate);
	inline float gradient(float x,ACTIVATION activate);

public:
	ACTIVATION mActivate;
	int mNumInputNodes;
	int mNumNodes;
	float **mWeights;
	float *mOutputs;
	float *mDelta;
};

#endif
