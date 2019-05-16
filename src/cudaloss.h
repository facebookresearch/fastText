#pragma once

#include <cudnn.h>
#include "cublas_v2.h"
#include "loss.h"

namespace fasttext {

class CudaState : public Model::State {
 public:
   CudaState(int32_t hiddenSize, int32_t outputSize, int32_t seed);
   ~CudaState();
  
   // CUDA vars
   real* d_hidden_;
   real* d_output_;
   real* d_softmax_output_;
   real* d_output_diff_;
   real* d_grad_;
   real* d_lossValue_;
   cudaStream_t stream_;
   cudnnHandle_t cudnn_;
   cudnnTensorDescriptor_t cudnn_output_desc_;
   cublasHandle_t cublas_;  
};

class CudaSoftmaxLoss : public SoftmaxLoss {
 public:
  explicit CudaSoftmaxLoss(std::shared_ptr<Matrix>& wi, std::shared_ptr<Matrix>& wo);
  virtual ~CudaSoftmaxLoss();

  virtual bool init();
  virtual void shutdown();

  virtual real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop);

 protected:
  void cudaforward(CudaState& batchState, int32_t target, real lr, bool backprop, real& lossValue, Vector& grad);
  void compare(const Model::State& CPUState, const CudaState& GPUState, bool CmpWo, bool CmpSoftmaxOutput);

 protected: 
  static real* d_wo_;
  std::shared_ptr<Matrix> wi_;
};

} // namespace fasttext
