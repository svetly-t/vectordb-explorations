#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <vector>
#include <queue>

#include <math.h>
#include <stdlib.h>

/* Returns 000..000 if the bool is false, else return 111...111 */
inline uint32_t ZeroIfTrue(bool stmt) {
  return ~(!stmt - 1ull);
}

class Vector {
 private:
  std::vector<float> data_;
  size_t size_;

 public:
  Vector(size_t size) {
    if (!size) throw std::runtime_error("cannot initialize Vector with size 0");
    data_.reserve(size);
    size_ = size;
  }

  Vector(std::vector<float> init) {
    if (!init.size()) throw std::runtime_error("cannot initialize Vector with size 0");
    size_ = init.size();
    data_ = std::move(init);
  }

  ~Vector() {
    if (!size_) return;
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

  /* For debug-printing */
  std::string Print() {
    std::string out = "";
    for (float f : data_)
      out += std::to_string(f) + " ";
    return out;
  }
};

class Hnsw {
 public:
  struct Node {
    Vector vec;
    std::vector<Node *> neighbors;
    Node *next_layer; /* Points to the /same/ node in the next layer of the layered graph */

    Node(size_t size) : vec(size) {}

    Node(std::vector<float> v) : vec(v) {}

    Node(Vector v) : vec(v) {}

    std::string Print() { return vec.Print(); }
  };

  Hnsw(size_t efConstruction, size_t layers, size_t m);

  /* Implement this */
  void Insert(Node node, size_t l);
  /******************/

  Node *FindNn(const Node &node);
 private:
  size_t efConstruction_ = 0; /* Number of NN to use as entry points when descending to next layer */
  size_t mL_ = 0; /* Maximum depth */
  size_t m_ = 0; /* Number of NN to connect to when layer <= l */
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

void Hnsw::Insert(Node node, size_t l) {
  /* Let first entry node be the first node in the top layer */
  Node *entry_node = &layers_[mL_ - 1].front();

  /* Step 1: above l */
  for (size_t layer = mL_ - 1; layer > l; --layer) {
    Node *next_entry_node = entry_node;

    /* Greedy search over all neighbors: find the one that's closest to the query */
    do {
      for (const auto &nd : entry_node->neighbors)
        if (nd->vec.Distance(node.vec) < next_entry_node->vec.Distance(node.vec))
          next_entry_node = nd;

    } while (next_entry_node != entry_node); /* break if there are no neighbor nodes closer than entry_... */
    entry_node = next_entry_node->next_layer;
  }

  /* Step 2: l.e.q. l */
  auto cmp = [&node](Node *left, Node *right) {
    return left->vec.Distance(node.vec) > right->vec.Distance(node.vec);
  };
  std::priority_queue<Node *, std::vector<Node *>, decltype(cmp)> closest_nodes(cmp);
  std::vector<Node *> entry_points;
  
  /* Start with our closest node from the upper layers */
  entry_points.push_back(entry_node);

  /* Node on layer above us */
  Node *last_layer = nullptr;
  
  for (size_t layer = l; layer >= 0; --layer) {
    /* Add the new node to this layer in the hnsw collection */
    layers_[l].emplace_back(std::move(node));
    Node *new_node = &layers_[l].back();

    if (last_layer != nullptr)
      last_layer->next_layer = new_node;
    
    last_layer = new_node;

    /* Get all the closest nodes starting from the entry points */
    for (Node *n = entry_points.back(); !entry_points.empty(); entry_points.pop_back()) {
      Node *next_n = n;
      /* Greedy search over all neighbors: find the one that's closest to the query */
      do {
        for (const auto &nd : n->neighbors)
          if (nd->vec.Distance(node.vec) < next_n->vec.Distance(node.vec)) {
            next_n = nd;
            closest_nodes.push(nd);
          }
      } while (next_n != n); /* break if there are no neighbor nodes closer than n */
    }

    /* Make links */
    for (size_t e = 0; e < efConstruction_; ++e) {
      if (closest_nodes.empty()) break;

      Node *nn = closest_nodes.top();
      closest_nodes.pop();

      entry_points.push_back(nn);

      if (e >= m_) continue;

      /* Only add edges if we're less than m, else just add entry points*/

      nn->neighbors.push_back(new_node);
      new_node->neighbors.push_back(nn);
    }
  }
}

void PriorityQueueTest() {
  Hnsw::Node compare_to(3);
  compare_to.vec.Set(0, 1.0);
  compare_to.vec.Set(1, 2.0);
  compare_to.vec.Set(2, 3.0);

  /**
   * Priority queue by default sorts things from biggest to smallest
   * But we want the lowest distance to be the first one that shows up.
   */
  auto cmp = [&compare_to](Hnsw::Node *left, Hnsw::Node *right){
    return left->vec.Distance(compare_to.vec) >
           right->vec.Distance(compare_to.vec);
  };
  std::priority_queue<Hnsw::Node *, std::vector<Hnsw::Node *>, decltype(cmp)> q(cmp);

  Hnsw::Node n1(3);
  n1.vec.Set(0, 2.0);
  n1.vec.Set(1, 4.0);
  n1.vec.Set(2, 6.0);

  Hnsw::Node n2(3);
  n2.vec.Set(0, 4.0);
  n2.vec.Set(1, 8.0);
  n2.vec.Set(2, 12.0);

  q.push(&n2);
  q.push(&n1);

  std::cout << "compare_to: " << compare_to.Print() << std::endl;
  std::cout << "top element: " << q.top()->Print() << std::endl;
  q.pop();
  std::cout << "next element: " << q.top()->Print() << std::endl;
  q.pop();

  return;
}

int main(int argc, char **argv) {
  Hnsw hnsw(5, 5, 2);

  /* Insert a bunch of RGB colors and find the most similar one */

  srand((uint64_t)argv[0]);

  for (size_t c = 0; c < 256; ++c) {
    float r = rand() % 256;
    float g = rand() % 256;
    float b = rand() % 256;
    Hnsw::Node n({ r, g, b });
  }
}