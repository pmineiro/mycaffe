#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedBatchAverageLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const int dim = (*top)[0]->count();
  const int num = bottom[0]->num();
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();

  caffe_gpu_set(dim, Dtype(0), top_data);
  for (int i = 0; i < num; ++i) {
    caffe_gpu_axpy(dim, bottom_data0[i] / num, bottom_data1 + i*dim, top_data);
  }
}

template <typename Dtype>
void WeightedBatchAverageLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  const int dim = top[0]->count();
  const int num = (*bottom)[0]->num();
  const Dtype* bottom_data0 = (*bottom)[0]->cpu_data();
  const Dtype* bottom_data1 = (*bottom)[1]->gpu_data();
  Dtype* bottom_diff0 = (*bottom)[0]->mutable_cpu_diff();
  Dtype* bottom_diff1 = (*bottom)[1]->mutable_gpu_diff();
  const Dtype* top_diff = top[0]->gpu_diff();

  // f (\sum_i w_i x_i)
  // (df/dw_i) = (df(v)/dv|_{v=\sum_i w_i x_i}) * x_i
  // (df/dx_i) = (df(v)/dv|_{v=\sum_i w_i x_i}) * w_i

  for (int i = 0; i < num; ++i) {
    if (propagate_down[0]) {
      caffe_gpu_dot (dim, bottom_data1 + i*dim, top_diff, bottom_diff0 + i);
    }

    if (propagate_down[1]) {
#define caffe_gpu_copy caffe_copy
      caffe_gpu_copy (dim, top_diff, bottom_diff1 + i*dim);
      caffe_gpu_scal (dim, bottom_data0[i] / num, bottom_diff1 + i*dim);
    }
  }

  if (propagate_down[0]) {
    caffe_scal (num, Dtype(1)/num, bottom_diff0);
  }
}

INSTANTIATE_CLASS(WeightedBatchAverageLayer);

}  // namespace caffe
