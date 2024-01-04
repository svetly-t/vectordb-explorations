#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <vector>
#include <queue>

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

  size_t Size() { return size_; }

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

/** Interesting stuff below **/

struct Node {
  Vector vec;
  std::vector<Node *> neighbors;

  /**
   * Kind of implementation-specific, but this points to
   * the /same/ node in the next layer of the layered graph
   */
  std::list<Node>::iterator next_layer; 

  Node(size_t size) : vec(size) {}
  
  std::string Print() {
    std::string out = "";
    for (size_t s = vec.Size(), i = 0; i < s; ++i)
      out += std::to_string(vec.Get(i)) + " ";
    return out;
  }
};

class Hnsw {
 public:
  Hnsw(size_t efConstruction, size_t layers, size_t m);

  /* Implement this */
  void Insert(Node node, size_t l);
  /******************/

  Node *FindNn(const Node &node);
 private:
  /* Number of NN to use as entry points when descending to next layer */
  size_t efConstruction_ = 0;
  /* Maximum depth */
  size_t mL_ = 0;
  /* Number of NN to connect to when layer < l */
  size_t m_ = 0;
  /**
   * std::vector from level 0 to mL
   * Each entry in the vector is a list of Nodes
   */
  std::vector<std::list<Node>> layers_;
};

Hnsw::Hnsw(size_t efConstruction, size_t layers, size_t m) {
  m_ = m;
  mL_ = layers;
  efConstruction_ = efConstruction;
  layers_.reserve(layers);
}

void PriorityQueueTest() {
  Node compare_to(3);
  compare_to.vec.Set(0, 1.0);
  compare_to.vec.Set(1, 2.0);
  compare_to.vec.Set(2, 3.0);

  /**
   * Priority queue by default sorts things from biggest to smallest
   * But we want the lowest distance to be the first one that shows up.
   */
  auto cmp = [&compare_to](Node *left, Node *right){
    return left->vec.Distance(compare_to.vec) >
           right->vec.Distance(compare_to.vec);
  };
  std::priority_queue<Node *, std::vector<Node *>, decltype(cmp)> q(cmp);

  Node n1(3);
  n1.vec.Set(0, 2.0);
  n1.vec.Set(1, 4.0);
  n1.vec.Set(2, 6.0);

  Node n2(3);
  n2.vec.Set(0, 4.0);
  n2.vec.Set(1, 8.0);
  n2.vec.Set(2, 12.0);

  q.push(&n2);
  q.push(&n1);

  std::cout << "compare_to: " << compare_to.Print() << std::endl;
  std::cout << "top element: " << q.top()->Print() << std::endl;
  q.pop();
  std::cout << "next element: " << q.top()->Print() << std::endl;

  return;
}

int main(int argc, char **argv) {
  PriorityQueueTest();
}