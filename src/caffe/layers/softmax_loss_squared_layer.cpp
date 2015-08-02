#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxWithLossDetailLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  softmax_bottom_vec_.clear();
  softmax_bottom_vec_.push_back(bottom[0]);
  softmax_top_vec_.clear();
  softmax_top_vec_.push_back(&prob_);
  softmax_layer_->SetUp(softmax_bottom_vec_, &softmax_top_vec_);
}

template <typename Dtype>
void SoftmaxWithLossDetailLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  (*top)[1]->Reshape(1, 1, 1, 1);
  // NB: bottom[0].height == bottom[0].width == 1 for inner product layer
  // otherwise this looks really weird
  (*top)[2]->Reshape(bottom[0]->num(),
                     1,
                     bottom[0]->height(),
                     bottom[0]->width());
  softmax_layer_->Reshape(softmax_bottom_vec_, &softmax_top_vec_);
  if (top->size() >= 4) {
    // softmax output
    (*top)[3]->ReshapeLike(*bottom[0]);
  }
}

template <typename Dtype>
void SoftmaxWithLossDetailLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  // The forward pass computes the softmax prob values.
  softmax_layer_->Forward(softmax_bottom_vec_, &softmax_top_vec_);
  const Dtype* prob_data = prob_.cpu_data();
  Dtype* loss_data = (*top)[2]->mutable_cpu_data();
  const Dtype* label = bottom[1]->cpu_data();
  int num = prob_.num();
  int dim = prob_.count() / num;
  int spatial_dim = prob_.height() * prob_.width();
  // NB: spatial_dim == 1 for inner product layer
  // otherwise this looks really weird
  Dtype loss = 0;
  Dtype losssquared = 0;
  for (int i = 0; i < num; ++i) {
    for (int j = 0; j < spatial_dim; j++) {
      Dtype l = -log(std::max(prob_data[i * dim +
          static_cast<int>(label[i * spatial_dim + j]) * spatial_dim + j],
                           Dtype(FLT_MIN)));
      loss_data[i * spatial_dim + j] = l;
      loss += l;
      losssquared += l*l;
    }
  }
  (*top)[0]->mutable_cpu_data()[0] = loss / num / spatial_dim;
  (*top)[1]->mutable_cpu_data()[0] = losssquared / num / spatial_dim;
  if (top->size() >= 4) {
    (*top)[3]->ShareData(prob_);
  }
}

template <typename Dtype>
void SoftmaxWithLossDetailLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down[1]) {
    LOG(FATAL) << this->type_name()
               << " Layer cannot backpropagate to label inputs.";
  }
  if (propagate_down[0]) {
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const Dtype* prob_data = prob_.cpu_data();
    caffe_copy(prob_.count(), prob_data, bottom_diff);
    const Dtype* label = (*bottom)[1]->cpu_data();
    const Dtype* loss_data = top[2]->cpu_data();
    int num = prob_.num();
    int dim = prob_.count() / num;
    int spatial_dim = prob_.height() * prob_.width();
    for (int i = 0; i < num; ++i) {
      for (int j = 0; j < spatial_dim; ++j) {
        bottom_diff[i * dim + static_cast<int>(label[i * spatial_dim + j])
            * spatial_dim + j] -= 1;
      }
      // TODO: this _only_ works if spatial_dim == 1
      caffe_scal(dim, loss_data[i], bottom_diff + i*dim);
    }
    // Scale gradient
    const Dtype loss_weight = top[0]->cpu_diff()[0];
    caffe_scal(prob_.count(), loss_weight / num / spatial_dim, bottom_diff);
  }
}


#ifdef CPU_ONLY
STUB_GPU(SoftmaxWithLossDetailLayer);
#endif

INSTANTIATE_CLASS(SoftmaxWithLossDetailLayer);

}  // namespace caffe
