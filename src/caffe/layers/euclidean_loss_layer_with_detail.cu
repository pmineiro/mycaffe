#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void EuclideanLossWithDetailLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  int count = bottom[0]->count();
  caffe_gpu_sub(
      count,
      bottom[0]->gpu_data(),
      bottom[1]->gpu_data(),
      diff_.mutable_gpu_data());
  Dtype loss = 0;
  Dtype* loss_data = (*top)[1]->mutable_cpu_data();
  const Dtype* diff_data = diff_.gpu_data();
  int spatial_dim = bottom[0]->channels() * bottom[0]->height() * bottom[0]->width();
  for (int i = 0; i < bottom[0]->num(); ++i)
    {
      caffe_gpu_dot(spatial_dim, 
                    diff_data + i*spatial_dim, 
                    diff_data + i*spatial_dim, 
                    loss_data + i);
      loss += loss_data[i];
    }
  loss = loss / bottom[0]->num() / Dtype(2);
  (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossWithDetailLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, vector<Blob<Dtype>*>* bottom) {
  for (int i = 0; i < 2; ++i) {
    if (propagate_down[i]) {
      const Dtype sign = (i == 0) ? 1 : -1;
      const Dtype alpha = sign * top[0]->cpu_diff()[0] / (*bottom)[i]->num();
      const Dtype* loss_data = top[1]->cpu_data();
      const Dtype* diff_data = diff_.gpu_data();
      Dtype* bottom_diff_data = (*bottom)[i]->mutable_gpu_diff();
      int spatial_dim = (*bottom)[i]->channels() * (*bottom)[i]->height() * (*bottom)[i]->width();

      for (int j = 0; j < (*bottom)[i]->num(); ++j)
        {
          caffe_gpu_axpby(
              spatial_dim,                         // count
              alpha * loss_data[j],                // alpha
              diff_data + j * spatial_dim,         // a
              Dtype(0),                            // beta
              bottom_diff_data + j * spatial_dim); // b
        }
    }
  }
}

INSTANTIATE_CLASS(EuclideanLossWithDetailLayer);

}  // namespace caffe
