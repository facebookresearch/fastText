#include <iostream>
#include <mutex>
#include "cudaloss.h"
#include "matrix.h"
#include "densematrix.h"
#include "vector.h"

namespace fasttext {
static const float one = 1.0;
static const float zero = 0.0;
static const real epsilon = 0.00001f;
real* CudaSoftmaxLoss::d_wo_;

#define CUDA_CHECK(error) { \
  if (error!=cudaSuccess){ \
    std::cerr<<"CUDA ERROR "<< cudaGetErrorName(error) << " in file "  << __FILE__ << " line " <<__LINE__<< std::endl; \
    exit(0); \
  }  \
}

static const char* cublasGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
    default:
      return "UNKNOWN CUBLAS ERROR";
  }
}

#define CUBLAS_CHECK(err) \
{ \
  if (CUBLAS_STATUS_SUCCESS != err) { \
    std::cerr<<"CUBLAS ERROR "<< cublasGetErrorEnum(err) << " in file "  << __FILE__ << " line " <<__LINE__<< std::endl; \
    exit(0); \
  } \
}

CudaState::CudaState(int32_t hiddenSize, int32_t outputSize, int32_t seed, uint32_t batchSize):Model::State(hiddenSize, outputSize, seed),
	        targets(batchSize), lrs(batchSize), hiddens(batchSize, Vector(hiddenSize)), inputs(batchSize), batchIndex(0), maxBatchSize(batchSize) {
  int64_t M = outputSize;
  int64_t N = hiddenSize;
  CUDA_CHECK(cudaMalloc((void**)&d_hidden_, batchSize*N*sizeof(real)));
  CUDA_CHECK(cudaMalloc((void**)&d_output_, M*sizeof(real)));
  CUDA_CHECK(cudaMalloc((void**)&d_softmax_output_, M*sizeof(real)));
  CUDA_CHECK(cudaMalloc((void**)&d_output_diff_, M*sizeof(real)));
  CUDA_CHECK(cudaMalloc((void**)&d_grads_, batchSize*N*sizeof(real)+batchSize*sizeof(real)));  // size of grads + loss
  d_lossValues_ = d_grads_ + batchSize*N;

  stream_ = cudaStreamPerThread;
  cudnnCreate(&cudnn_);
  cudnnCreateTensorDescriptor(&cudnn_output_desc_);
  cudnnSetTensor4dDescriptor(cudnn_output_desc_, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, 1, 1, M);
  cudnnSetStream(cudnn_, stream_);
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_));
}

CudaState::~CudaState() {
  CUDA_CHECK(cudaFree(d_hidden_));
  CUDA_CHECK(cudaFree(d_output_));
  CUDA_CHECK(cudaFree(d_softmax_output_));
  CUDA_CHECK(cudaFree(d_output_diff_));
  CUDA_CHECK(cudaFree(d_grads_));
  cudnnDestroyTensorDescriptor(cudnn_output_desc_);
  cudnnDestroy(cudnn_);
  CUBLAS_CHECK(cublasDestroy(cublas_));
}

void CudaState::addToBatch(int32_t target, real lr, const std::vector<int32_t>& input) {
  targets[batchIndex] = target;
  lrs[batchIndex] = lr;
  hiddens[batchIndex] = hidden;
  inputs[batchIndex] = input;
  batchIndex++;
}

CudaSoftmaxLoss::CudaSoftmaxLoss(std::shared_ptr<Matrix>& wi, std::shared_ptr<Matrix>& wo, bool normalizeGradient):SoftmaxLoss(wo), wi_(wi), normalizeGradient_(normalizeGradient) {
}

CudaSoftmaxLoss::~CudaSoftmaxLoss() {
}

bool CudaSoftmaxLoss::init() {
  // Copy wo from host to device
  int64_t m = wo_->size(0);
  int64_t n = wo_->size(1);
  std::vector<real> tmpwo(m*n);
  real* pBegin = tmpwo.data();
  for( int64_t i=0; i<m; i++ ) {
    Vector v(n);
    wo_->addRowToVector(v, i);
    memcpy(pBegin+i*n, v.data(), n*sizeof(real));
  }

  CUDA_CHECK(cudaMalloc((void**)&d_wo_, m*n*sizeof(real)));
  CUDA_CHECK(cudaMemcpy(d_wo_, pBegin, m*n*sizeof(real), cudaMemcpyHostToDevice));
  return true;
}

void CudaSoftmaxLoss::shutdown() {
  // Copy wo from device back to host
  int64_t m = wo_->size(0);
  int64_t n = wo_->size(1);
  std::vector<real> tmpwo(m*n);
  CUDA_CHECK(cudaMemcpy(tmpwo.data(), d_wo_, m*n*sizeof(real), cudaMemcpyDeviceToHost));
  real* pBegin = tmpwo.data();
  for( int64_t i=0; i<m; i++ ) {
    Vector clear(n);
    wo_->addRowToVector(clear, i);
    wo_->addVectorToRow(clear, i, -1.0);

    Vector add(n);
    memcpy(add.data(), pBegin+i*n, n*sizeof(real));
    wo_->addVectorToRow(add, i, 1.0);
  }
  CUDA_CHECK(cudaFree(d_wo_));
}

real CudaSoftmaxLoss::forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop) {
  throw std::runtime_error("forward is not supported in CudaSoftmaxLoss");
}

void CudaSoftmaxLoss::computeOutput(Model::State& state) const {
}

bool CudaSoftmaxLoss::batchforward_enabled() const {
  return true;
}

__global__
void CudacomputeDiff(real* softmax_output, size_t output_n, real* output_diff, real* loss, int32_t target, real lr) {
  int output_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if( threadIdx.x==0 && blockIdx.x==0 ) {
    *loss = softmax_output[target];
  }

  if( output_idx < output_n ) {
    real label = (output_idx==target)?1.0:0.0;
    output_diff[output_idx] = lr * (label - softmax_output[output_idx]);
  }
}

void CudaSoftmaxLoss::forward2batch(int32_t target, Model::State& state, real lr, bool backprop, bool normalizeGradient, const std::vector<int32_t>& input) {
  CudaState& batchState = static_cast<CudaState&>(state);

#ifdef FASTTEXT_CUDA_DEBUG
  Model::State CpuState(state);
  compare(CpuState, batchState, true, false);
  SoftmaxLoss::computeOutput(CpuState);
  if (backprop) {
    int32_t osz = wo_->size(0);
    for (int32_t i = 0; i < osz; i++) {
      real label = (i == target) ? 1.0 : 0.0;
      real alpha = lr * (label - CpuState.output[i]);
      CpuState.grad.addRow(*wo_, i, alpha);
      wo_->addVectorToRow(CpuState.hidden, i, alpha);
    }
  }
  real cpuloss = -SoftmaxLoss::log(CpuState.output[target]);
#endif

  batchState.addToBatch(target, lr, input);
  if( batchState.batchIndex==batchState.maxBatchSize ) {
    flush(batchState, backprop);
  }

#ifdef FASTTEXT_CUDA_DEBUG
  if( backprop )
    compare(CpuState, batchState, false, true);
#endif  
}

void CudaSoftmaxLoss::flush(Model::State& state, bool backprop) {
  CudaState& batchState = static_cast<CudaState&>(state);
  std::vector<real> lossValues(batchState.batchIndex);
  std::vector<Vector> grads(batchState.batchIndex, Vector(batchState.grad.size()));
  batchforward(batchState, batchState.batchIndex, backprop, lossValues, grads);
  for (const auto& lossValue : lossValues) {
    batchState.incrementNExamples(lossValue);
  }
  int idx = 0;
  for (auto& grad : grads) {
    if (normalizeGradient_) {
      grad.mul(1.0 / batchState.inputs[idx].size());
    }
    for (auto it = batchState.inputs[idx].cbegin(); it != batchState.inputs[idx].cend(); ++it) {
      wi_->addVectorToRow(grad, *it, 1.0);
    }
    idx++;
  }
  batchState.batchIndex = 0;
}

void CudaSoftmaxLoss::compare(const Model::State& CPUState, const CudaState& GPUState, bool CmpWo, bool CmpSoftmaxOutput) {
  if( CmpWo ) {
    int64_t m = wo_->size(0);
    int64_t n = wo_->size(1);
    DenseMatrix* wo = dynamic_cast<DenseMatrix*>(wo_.get());
    std::vector<real> tmpwo(m*n);
    CUDA_CHECK(cudaMemcpy(tmpwo.data(), d_wo_, m*n*sizeof(real), cudaMemcpyDeviceToHost));
    for( int64_t i=0; i<m; i++ ) {
      for( int64_t j=0; j<n; j++ ) {
        if( fabs(tmpwo[i*n+j]-wo->at(i,j))>epsilon )
          printf("\nwo[%ld,%ld] not match %f %f\n", i, j, tmpwo[i*n+j], wo->at(i,j));
      }
    }
  }
  if( CmpSoftmaxOutput ) {
    int64_t m = wo_->size(0);
    std::vector<real> tmpSoftMax(m);
    CUDA_CHECK(cudaMemcpy(tmpSoftMax.data(), GPUState.d_softmax_output_, m*sizeof(real), cudaMemcpyDeviceToHost));
    for( int64_t i=0; i<m; i++ ) {
      if( fabs(tmpSoftMax[i]-CPUState.output[i])>epsilon )
	printf("\nsoftmax [%ld] not match %f %f\n", i, tmpSoftMax[i], CPUState.output[i]);
    }
  }
}

void CudaSoftmaxLoss::batchforward(
      CudaState& batchState,
      uint32_t batchSize,
      bool backprop,
      std::vector<real>& lossValues,
      std::vector<Vector>& grads) {
  int M = wo_->size(0);  // labels
  int N = wo_->size(1);  // dims

  // Copy hiddens -> d_hidden_
  std::vector<real> h_hiddens(batchSize*N);
  real* pHidden = h_hiddens.data();
  for(uint32_t i=0; i<batchSize; i++ ) {
    memcpy(pHidden+i*N, batchState.hiddens[i].data(), N*sizeof(real));
  }
  CUDA_CHECK(cudaMemcpy(batchState.d_hidden_, pHidden, batchSize*N*sizeof(real), cudaMemcpyHostToDevice));
  h_hiddens.clear();

  for(uint32_t i=0; i<batchSize; i++ ) {
    // compute output
    CUBLAS_CHECK(cublasSgemv(batchState.cublas_, CUBLAS_OP_T,
      N, M,
      &one,
      d_wo_, N,
      batchState.d_hidden_+N*i, 1,
      &zero,
      batchState.d_output_, 1));

    // compute softmax
    cudnnSoftmaxForward(batchState.cudnn_, cudnnSoftmaxAlgorithm_t::CUDNN_SOFTMAX_ACCURATE, cudnnSoftmaxMode_t::CUDNN_SOFTMAX_MODE_INSTANCE,
      &one, batchState.cudnn_output_desc_, batchState.d_output_,
      &zero, batchState.cudnn_output_desc_, batchState.d_softmax_output_);  

    // compute loss
    CudacomputeDiff<<<(M+255)/256, 256, 0, batchState.stream_>>>(
      batchState.d_softmax_output_,
      M,
      batchState.d_output_diff_,
      batchState.d_lossValues_ + i,
      batchState.targets[i], batchState.lrs[i]);

    if( backprop ) {
      // compute grad
      CUBLAS_CHECK(cublasSgemv(batchState.cublas_, CUBLAS_OP_T,
        M, N,
        &one,
        d_wo_, M,
        batchState.d_output_diff_, 1,
        &zero,
        batchState.d_grads_+N*i, 1));

      // update wo
      CUBLAS_CHECK(cublasSger(batchState.cublas_,
        N, M,
        &one,
        batchState.d_hidden_, 1,
        batchState.d_output_diff_, 1,
        d_wo_, N));
    }
  }

  cudaStreamSynchronize(batchState.stream_);

  // Copy d_lossValues_ -> lossValues, d_grads_ -> grads
  std::vector<real> h_grads_loss(batchSize*N + batchSize);
  CUDA_CHECK(cudaMemcpy(h_grads_loss.data(), batchState.d_grads_, batchSize*N*sizeof(real)+batchSize*sizeof(real), cudaMemcpyDeviceToHost));
  real* pGrad = h_grads_loss.data();
  for(uint32_t i=0; i<batchSize; i++ ) {
    memcpy(grads[i].data(), pGrad+i*N, N*sizeof(real));
    lossValues[i] = -Loss::log(h_grads_loss[batchSize*N+i]);
  }
}

} // namespace fasttext
