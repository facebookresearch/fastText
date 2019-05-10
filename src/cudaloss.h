#pragma once

#include <cudnn.h>
#include "cublas_v2.h"
#include "loss.h"

namespace fasttext {

class CudaState : public Model::State {
 public:
   CudaState(int32_t hiddenSize, int32_t outputSize, int32_t seed, uint32_t batchSize);
   ~CudaState();
   void addToBatch(int32_t target, real lr, const std::vector<int32_t>& input);
   std::vector<int32_t> targets;
   std::vector<real> lrs;
   std::vector<Vector> hiddens;
   std::vector< std::vector<int32_t> > inputs;
   uint32_t batchIndex;
   uint32_t maxBatchSize;
  
   // CUDA
   real* d_hidden_;
   real* d_output_;
   real* d_softmax_output_;
   real* d_output_diff_;
   real* d_grads_;
   real* d_lossValues_;
   cudaStream_t stream_;
   cudnnHandle_t cudnn_;
   cudnnTensorDescriptor_t cudnn_output_desc_;
   cublasHandle_t cublas_;  
};

class CudaSoftmaxLoss : public SoftmaxLoss {
 public:
  explicit CudaSoftmaxLoss(std::shared_ptr<Matrix>& wi, std::shared_ptr<Matrix>& wo, bool normalizeGradient);
  virtual ~CudaSoftmaxLoss();

  virtual bool init();
  virtual void shutdown();

  virtual void flush(Model::State& state, bool backprop = true);
  virtual bool batchforward_enabled() const;
  virtual void forward2batch(int32_t target, Model::State& state, real lr, bool backprop, bool normalizeGradient, const std::vector<int32_t>& input);

  void batchforward(
      CudaState& batchState,
      uint32_t batchSize,
      bool backprop,
      std::vector<real>& lossValues,
      std::vector<Vector>& grads);

 private:
  void compare(const Model::State& CPUState, const CudaState& GPUState, bool CmpWo, bool CmpSoftmaxOutput);  
  virtual real forward(
      const std::vector<int32_t>& targets,
      int32_t targetIndex,
      Model::State& state,
      real lr,
      bool backprop);
  virtual void computeOutput(Model::State& state) const;

 protected:
  static real* d_wo_;
  std::shared_ptr<Matrix> wi_;
  bool normalizeGradient_;
};

} // namespace fasttext
