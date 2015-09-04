#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void WeightedBatchAverageLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom[0]->num(), bottom[1]->num());
  CHECK_EQ(bottom[0]->channels(), 1);
  CHECK_EQ(bottom[0]->height(), 1);
  CHECK_EQ(bottom[0]->width(), 1);
  (*top)[0]->Reshape(1, bottom[1]->channels(), bottom[1]->height(), bottom[1]->width());
}

template <typename Dtype>
void WeightedBatchAverageLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  const int dim = (*top)[0]->count();
  const int num = bottom[0]->num();
  const Dtype* bottom_data0 = bottom[0]->cpu_data();
  const Dtype* bottom_data1 = bottom[1]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();

  caffe_set(dim, Dtype(0), top_data);
  for (int i = 0; i < num; ++i) {
    caffe_axpy(dim, bottom_data0[i] / num, bottom_data1 + i*dim, top_data);
  }
}

template <typename Dtype>
void WeightedBatchAverageLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {

  const int dim = top[0]->count();
  const int num = (*bottom)[0]->num();
  const Dtype* bottom_data0 = (*bottom)[0]->cpu_data();
  const Dtype* bottom_data1 = (*bottom)[1]->cpu_data();
  Dtype* bottom_diff0 = (*bottom)[0]->mutable_cpu_diff();
  Dtype* bottom_diff1 = (*bottom)[1]->mutable_cpu_diff();
  const Dtype* top_diff = top[0]->cpu_diff();

  // f (\sum_i w_i x_i)
  // (df/dw_i) = (df(v)/dv|_{v=\sum_i w_i x_i}) * x_i
  // (df/dx_i) = (df(v)/dv|_{v=\sum_i w_i x_i}) * w_i

  for (int i = 0; i < num; ++i) {
    if (propagate_down[0]) {
      bottom_diff0[i] = caffe_cpu_dot (dim, bottom_data1 + i*dim, top_diff) / num;
    }

    if (propagate_down[1]) {
      caffe_copy (dim, top_diff, bottom_diff1 + i*dim);
      caffe_scal (dim, bottom_data0[i] / num, bottom_diff1 + i*dim);
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(WeightedBatchAverageLayer);
#endif

INSTANTIATE_CLASS(WeightedBatchAverageLayer);

}  // namespace caffe
