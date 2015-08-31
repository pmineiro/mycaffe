#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/common_layers.hpp"

namespace caffe {

template <typename Dtype>
void GradientScalerLayer<Dtype>::LayerSetUp(
     const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::LayerSetUp(bottom, top);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  NeuronLayer<Dtype>::Reshape(bottom, top);
  (*top)[0]->ReshapeLike(*bottom[0]);
  (*top)[1]->Reshape(1, 1, 1, 1);
}


template <typename Dtype>
void GradientScalerLayer<Dtype>::Forward_cpu(
     const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  (*top)[0]->ShareData(*bottom[0]);
}

template <typename Dtype>
void GradientScalerLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[0]) {
    const int count = (*bottom)[0]->count();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* coeffptr = top[1]->cpu_data();

    caffe_cpu_scale(count, coeffptr[0], top_diff, bottom_diff);
  }
}

#ifdef CPU_ONLY
STUB_GPU(GradientScalerLayer);
#endif

INSTANTIATE_CLASS(GradientScalerLayer);

}  // namespace caffe
