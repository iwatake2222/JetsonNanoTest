/*** Include ***/
#include <stdio.h>
#include <chrono>
#include <assert.h>
#include <string>
#include <algorithm>
#include <fstream>
#include <iterator>
#include <memory>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>
#include <NvInfer.h>
#include <NvOnnxParser.h>

using namespace nvinfer1;

/*** Definition ***/
#define CONVERT_ONNX_TO_TRT
// #define CONVERT_UFF_TO_TRT

/* Model information */
#define ONNX_MODEL_NAME "mobilenetv2-1.0.onnx"
#define TRT_MODEL_NAME  "mobilenetv2-1.0.trt"
#define LABEL_NAME      "synset.txt"
#define INPUT_W 224
#define INPUT_H 224
#define INPUT_C 3
#define OUTPUT_SCORE_SIZE (1000 * 1)
static const float PIXEL_MEAN[3] = {0.485f, 0.456f, 0.406f};
static const float PIXEL_STD[3] = {0.229f, 0.224f, 0.225f};
typedef enum {
	INDEX_INPUT = 0,
	INDEX_SCORE = 1,
	INDEX_NUM,
} MODEL_IO;

/*** Macro ***/
#define CHECK(status)                             \
	do                                            \
	{                                             \
		auto ret = (status);                      \
		if (ret != 0)                             \
		{                                         \
			printf("Cuda failure: ");             \
			abort();                              \
		}                                         \
	} while (0)

/* very very simple Logger class */
class Logger : public nvinfer1::ILogger
{
public:
	Logger(Severity severity = Severity::kWARNING)
		: reportableSeverity(severity)
	{
	}

	void log(Severity severity, const char* msg) override
	{
		if (severity > reportableSeverity) return;
		std::cerr << msg << std::endl;
	}

	Severity reportableSeverity;
};


/*** Global variable ***/
static Logger s_logger;


/***** Function *************************************************************/
static void readLabel(const char* filename, std::vector<std::string> &labels)
{
	std::ifstream ifs(filename);
	if (ifs.fail()) {
		printf("failed to read %s\n", filename);
		return;
	}
	std::string str;
	while(getline(ifs, str)) {
		labels.push_back(str);
	}
}

static void doInference(IExecutionContext& context, float* input, float* scores, int batchSize)
{
	assert(context.getEngine().getNbBindings() == INDEX_NUM);
	void* buffers[INDEX_NUM];

	/* create GPU buffers and a stream */
	CHECK(cudaMalloc(&buffers[INDEX_INPUT], batchSize * INPUT_W * INPUT_H * INPUT_C * sizeof(float)));
	CHECK(cudaMalloc(&buffers[INDEX_SCORE], batchSize * OUTPUT_SCORE_SIZE * sizeof(float)));

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));

	/* DMA the input to the GPU,  execute the batch asynchronously, and DMA it back */
	CHECK(cudaMemcpyAsync(buffers[INDEX_INPUT], input, batchSize * INPUT_W * INPUT_H * INPUT_C * sizeof(float), cudaMemcpyHostToDevice, stream));
	context.enqueue(batchSize, buffers, stream, nullptr);
	CHECK(cudaMemcpyAsync(scores, buffers[INDEX_SCORE], batchSize * OUTPUT_SCORE_SIZE * sizeof(float), cudaMemcpyDeviceToHost, stream));
	cudaStreamSynchronize(stream);

	/* release the stream and the buffers */
	cudaStreamDestroy(stream);
	CHECK(cudaFree(buffers[INDEX_INPUT]));
	CHECK(cudaFree(buffers[INDEX_SCORE]));
}

// https://devtalk.nvidia.com/default/topic/1049024/what-is-the-defaulat-output-format-of-the-jetson-board-camera/?offset=6#5324356
// sudo apt install v4l-utils
// v4l2-ctl -d /dev/video0 --list-formats-ext
static std::string get_tegra_pipeline(int width, int height, int fps)
{
	return "nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)" + std::to_string(width) + ", height=(int)" +
	std::to_string(height) + ", format=(string)NV12, framerate=(fraction)" + std::to_string(fps) +
	"/1 ! nvvidconv flip-method=2 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink";
}


int main(int argc, char** argv)
{
	/*** Initialize ***/
	/* read label */
	std::vector<std::string> labels;
	readLabel(LABEL_NAME, labels);

	/* create runtime and engine from model file */
	IRuntime* runtime = createInferRuntime(s_logger);
	ICudaEngine* engine = NULL;

#if defined CONVERT_ONNX_TO_TRT
	/* create a TensorRT model from the onnx model */
	IBuilder* builder = createInferBuilder(s_logger);
	nvinfer1::INetworkDefinition* network = builder->createNetwork();
	auto parser = nvonnxparser::createParser(*network, s_logger);

	if (!parser->parseFromFile(ONNX_MODEL_NAME, (int)nvinfer1::ILogger::Severity::kWARNING)) {
		printf("failed to parse onnx file");
		exit(EXIT_FAILURE);
	}

	builder->setMaxBatchSize(1);
	builder->setMaxWorkspaceSize(1 << 30);
	builder->setAverageFindIterations(8);
	builder->setMinFindIterations(8) ;
	builder->setFp16Mode(false);
	builder->setInt8Mode(false);

	engine = builder->buildCudaEngine(*network);

	parser->destroy();
	network->destroy();
	builder->destroy();

#if 1
	/* save serialized model */
	IHostMemory* trtModelStream = engine->serialize();
	std::ofstream ofs(TRT_MODEL_NAME, std::ios::out | std::ios::binary);
	ofs.write((char*)(trtModelStream->data()), trtModelStream->size());
	ofs.close();
	trtModelStream->destroy();
#endif
#elif defined CONVERT_UFF_TO_TRT
	// todo: parse from UFF
#else
	std::string buffer;
	std::ifstream stream(TRT_MODEL_NAME, std::ios::binary);

	if (stream) {
		stream >> std::noskipws;
		copy(std::istream_iterator<char>(stream), std::istream_iterator<char>(), back_inserter(buffer));
	}
	engine = runtime->deserializeCudaEngine(buffer.data(), buffer.size(), nullptr);
	stream.close();
#endif

	assert(engine != nullptr);
	IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);

	/*** run inference ***/
	long long inferenceTime = 0;
	int cnt = 0;
	float *input = new float[INPUT_W * INPUT_H * INPUT_C];
	std::vector<float> scores(OUTPUT_SCORE_SIZE);
	cv::VideoCapture cap;
	cv::Mat frame;
	cap.open(get_tegra_pipeline(640, 480, 30));

	while (cap.isOpened() && cap.read(frame)) {
		/* Pre-Process */
		cv::Mat imgResized;
		cv::resize(frame, imgResized, cv::Size(INPUT_W, INPUT_H));
		
		/* convert NHWC to NCHW */
		#pragma omp parallel for
		for (int c = 0; c < INPUT_C; c++) {
			for (int i = 0; i < INPUT_W * INPUT_H; i++) {
				// input[c * INPUT_W * INPUT_H + i] = (float)(imgResized.data[i * INPUT_C + c] / 255.0);
				input[c * INPUT_W * INPUT_H + i] = (float)  (((imgResized.data[i * INPUT_C + c] / 255.0) - PIXEL_MEAN[c]) / PIXEL_STD[c]);
			}
		}

		/* Inference */
		auto t0 = std::chrono::system_clock::now();
		doInference(*context, input, &scores[0], 1);
		auto t1 = std::chrono::system_clock::now();
		inferenceTime += std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

		cv::imshow("frame", frame);
		int key = cv::waitKey(1);
		if (key == 'q') break;

		/* Post-Process */
		int maxIndex = std::max_element(scores.begin(), scores.end()) - scores.begin();
		float maxScore = *std::max_element(scores.begin(), scores.end());
		double sumSoftMax = 0;
		for(auto score : scores) sumSoftMax += exp(score);
		printf("Result = %s: %d (%.3lf)\n", labels[maxIndex].c_str(), maxIndex, exp(maxScore) / sumSoftMax);
		cnt++;
	}
	printf("Inference time: %.2lf [msec]\n", double(inferenceTime) / cnt);

	/*** Finalize ***/
	delete[] input;
	context->destroy();
	engine->destroy();
	runtime->destroy();

	cv::destroyAllWindows();

	return 0;
}

