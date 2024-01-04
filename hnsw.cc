#include <iostream>
#include <stdexcept>
#include <vector>

#include <math.h>

/* Returns 000..000 if the bool is false, else return 111...111 */
inline uint32_t ZeroIfTrue(bool stmt) {
  return ~(!stmt - 1ull);
}

class Vector {
 private:
  float *data_ = nullptr;
  size_t size_ = 0;

 public:
  Vector(size_t size) {
    if (!size)
      throw std::runtime_error("cannot initialize Vector with size 0");
    data_ = new float[size];
    size_ = size;
  }

  ~Vector() {
    if (!size_)
      return;
    delete data_;
  }

  void Set(size_t idx, float v) {
    if (idx > size_) return;
    data_[idx] = v;
  }
  
  /**
   * To avoid branch and segfault we get the value at
   * idx % size_ but mask to zero if idx is g.t. size_
   */
  float Get(size_t idx) const {
    uint32_t mask = ZeroIfTrue(idx >= size_);
    uint32_t val = *(uint32_t *)(&data_[idx % size_]);
    val = mask & val;
    float ret = *(float *)&val;
    return ret;
  }

  float Distance(const Vector &o) const {
    float d = 0.0;
    for (size_t n = 0; n < size_; ++n) {
      float v1 = Get(n);
      float v2 = o.Get(n);
      float v = v1 - v2;
      d += v * v;
    }
    return sqrt(d);
  }
};

struct Node {
  Vector vec;
  std::vector<Node *> neighbors;

  Node(size_t size) : vec(size) {}
};

class Hnsw {
 public:
  void Insert(Node node);
  void FindNn(const Node &node);
 private:  
};

int main(int argc, char **argv) {
  Vector alpha(4);
  Vector beta(8);

  for (size_t n = 0; n < 8; ++n) {
    alpha.Set(n, 1.0);
    beta.Set(n, 1.0);
  }

  float dist = beta.Distance(alpha);
  std::cout << "distance: " << dist << std::endl;

  return 0;
}