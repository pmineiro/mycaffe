#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossWithDetailLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  (*top)[1]->Reshape(bottom[0]->num(), 1, 1, 1);
}

template <typename Dtype>
void EuclideanLossWithDetailLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_sub(
      count,
      bottom[0]->cpu_data(),
      bottom[1]->cpu_data(),
      diff_.mutable_cpu_data());
  Dtype loss = 0;
  Dtype* loss_data = (*top)[1]->mutable_cpu_data();
  const Dtype* diff_data = diff_.cpu_data();
  int spatial_dim = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  for (int i = 0; i < bottom[0]->num(); ++i)
    {
      loss_data[i] = caffe_cpu_dot(spatial_dim, 
                                   diff_data + i*spatial_dim,
                                   diff_data + i*spatial_dim);
      loss += loss_data[i];
    }

  loss = loss / bottom[0]->num() / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossWithDetailLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();
      const Dtype* loss_data = top[1]->cpu_data();
      const Dtype* diff_data = diff_.cpu_data();
      Dtype* bottom_diff_data = (*bottom)[i]->mutable_cpu_diff();
      int spatial_dim = (*bottom)[i]->channels() * (*bottom)[i]->height() * (*bottom)[i]->width();

      for (int j = 0; j < (*bottom)[i]->num(); ++j)
        {
          caffe_cpu_axpby(
              spatial_dim,                         // count
              alpha * loss_data[j],                // alpha
              diff_data + j * spatial_dim,         // a
              Dtype(0),                            // beta
              bottom_diff_data + j * spatial_dim); // b
        }
    }
  }
}

#ifdef CPU_ONLY
STUB_GPU(EuclideanLossWithDetailLayer);
#endif

INSTANTIATE_CLASS(EuclideanLossWithDetailLayer);

}  // namespace caffe
