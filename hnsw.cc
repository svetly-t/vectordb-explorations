#include <algorithm>
#include <iostream>
#include <list>
#include <stdexcept>
#include <string>
#include <vector>
#include <queue>

#include <math.h>
#include <stdlib.h>

#ifdef _WIN32
#include <Windows.h>
#include <Psapi.h>
#pragma comment( lib, "Psapi.lib" )
#endif

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
    data_.resize(size);
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
    /* Each node is on exactly one layer */
    size_t layer;
    size_t id;
     /* Points to the /same/ node in the next layer of the layered graph */
    Node *next_layer = nullptr;

    Node(size_t size) : vec(size) {}

    Node(std::vector<float> v) : vec(v) {}

    Node(Vector v) : vec(v) {}

    std::string Print() { return vec.Print(); }
  };

  Hnsw(size_t efConstruction, size_t layers, size_t m);

  size_t RandomLevel();

  void Insert(std::vector<float> vec, size_t l);
  void Insert(Node query, size_t l);

  std::vector<Node *> FindNearest(const Node &query, size_t neighbors); 

 private:
  size_t efConstruction_ = 0; /* Number of NN to use as entry points when descending to next layer */
  size_t l_ = 0; /* Number of layers */
  size_t m_ = 0; /* Number of NN to connect to when layer <= l */
  std::vector<std::list<Node>> layers_;

  std::vector<Node *> FindNn(const Node &quer, size_t l, size_t n, Node *entry);
};

Hnsw::Hnsw(size_t efConstruction, size_t layers, size_t m) {
  m_ = m;
  l_ = layers;
  efConstruction_ = efConstruction;
  layers_.resize(layers);
}

/* Pick a random level from zero to mL with logarithmic falloff */
size_t Hnsw::RandomLevel() {
  double uniform = (double)rand() / (double)RAND_MAX;
  double result = -log(uniform) * (1.0 / m_);
  if (result >= l_) result = l_;
  return result;
}

void Hnsw::Insert(Node query, size_t l) {
  std::vector<Node *> entry_points;
  
  /**** Step 1: greater than l. ****/
  size_t layer = l_ - 1;
  Node *entry = nullptr;
  for (; layer > l; --layer) {
    entry_points = FindNn(query, layer, 1, entry);
    entry = entry_points.size() ? entry_points.back()->next_layer : nullptr;
  }

  if (entry_points.size()) {
    entry_points.clear();
    entry_points.push_back(entry);
  }

  /**** Step 2: less than or equal to l. ****/

  /* Node on layer above us */
  Node *prev_layer = nullptr;
  Node *new_node = nullptr;

  for (; layer >= 0; --layer) {
    std::vector<Node *> closest;

    /* Get all the closest nodes starting from the entry points */
    if (entry_points.empty()) entry_points.push_back(nullptr);

    for (Node *entry = entry_points.back(); !entry_points.empty(); entry_points.pop_back()) {
      std::vector<Node *> closest_to_entry = FindNn(query, layer, efConstruction_, entry);
      closest.insert(closest.end(), closest_to_entry.begin(), closest_to_entry.end());
    }
    auto cmp = [&query](Node *l, Node *r) -> bool {
      return l->vec.Distance(query.vec) < r->vec.Distance(query.vec);
    };
    std::sort(closest.begin(), closest.end(), cmp);

    /* Add the node to the layer */
    layers_[layer].emplace_back(query);

    new_node = &layers_[layer].back();
    new_node->layer = layer;

    if (prev_layer != nullptr)
      prev_layer->next_layer = new_node;
    prev_layer = new_node;

    /* Make links */
    for (size_t e = 0; e < efConstruction_; ++e) {
      if (closest.size() <= e) break;

      Node *nn = closest[e];

      entry_points.push_back(nn->next_layer);

      if (e >= m_) continue;

      /* Only add edges if we're less than m, else just add entry points*/

      if (nn->layer != new_node->layer)
        std::cout << "layer mismatch!" << std::endl;

      nn->neighbors.push_back(new_node);
      new_node->neighbors.push_back(nn);
    }

    if (layer == 0) break;
  }
}

void Hnsw::Insert(std::vector<float> vec, size_t l) {
  Node nn(std::move(vec));
  Insert(std::move(nn), l);
}

/**
 * Returns a vector of the 'n' nearest neighbors to node on layer 'l',
 * starting from entry Node 'entry'.
 * 
 * If 'entry' is nullptr then we start searching from an arbitrary node
 * on layer 'l'.
 * 
 * May return fewer than 'n' nodes.
 */
std::vector<Hnsw::Node *> Hnsw::FindNn(const Node &query, size_t l, size_t n, Node *entry) {
  auto cmp = [&query](Node *l, Node *r) {
    return l->vec.Distance(query.vec) > r->vec.Distance(query.vec);
  };
  std::priority_queue<Node *, std::vector<Node *>, decltype(cmp)> closest_nodes(cmp);

  /**
   * No entry node specified, start at the beginning of the std::list for 'l'.
   */
  if (!entry) {
    if (layers_[l].empty()) return std::vector<Hnsw::Node *>();
    entry = &layers_[l].front();
  }
  
  closest_nodes.push(entry);

  for (Node *next = entry;;) {
    if (entry->neighbors.empty()) break;
    
    for (const auto &node : entry->neighbors)
      if (node->vec.Distance(query.vec) < next->vec.Distance(query.vec)) {
        closest_nodes.push(node);
        next = node;
      }

    if (next == entry) break;
    
    entry = next;
  }

  /**
   * SLOW -- convert the pqueue to a vector.
   * There may be a faster way to do this:
   * https://stackoverflow.com/questions/1185252/is-there-a-way-to-access-the-underlying-container-of-stl-container-adaptors
   */
  std::vector<Node *> res;
  for (; n > 0; --n) {
    if (closest_nodes.empty()) break;
    res.push_back(closest_nodes.top());
    closest_nodes.pop();
  }

  return res;
}

std::vector<Hnsw::Node *> Hnsw::FindNearest(const Node &query, size_t neighbors) {
  std::vector<Node *> points;
  Node *entry = nullptr;
  for (size_t layer = l_ - 1; layer > 0; --layer) {
    points = FindNn(query, layer, 1, entry);
    entry = points.size() ? points.back()->next_layer : nullptr;
  }
  points = FindNn(query, 0, neighbors, entry);
  return points;
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

/* Insert a bunch of RGB colors into a hnsw and find the most similar one */
void RgbTest(size_t num_colors) {
  Hnsw hnsw(5, 5, 2);

  srand(3);

  std::vector<Hnsw::Node> nodes;

  Hnsw::Node query(3);

  /* Add a bunch of random colors to the graph */
  for (size_t c = 0; c < num_colors; ++c) {
    float r = rand() % 256;
    float g = rand() % 256;
    float b = rand() % 256;
    size_t level = hnsw.RandomLevel();

    Hnsw::Node n({ r, g, b });
    n.id = c;
    nodes.push_back(n);
    hnsw.Insert(std::move(n), level);
  }

  /** 
   * Choose one of the colors and modify it slightly for our query.
   * We expect the color that we chose to be our nearest neighbor.
   */
  Hnsw::Node &neighbor = nodes[rand() % num_colors];

  query.vec.Set(0, neighbor.vec.Get(0));
  query.vec.Set(1, neighbor.vec.Get(1) + 1.0);
  query.vec.Set(2, neighbor.vec.Get(2) - 1.0);

  std::vector<Hnsw::Node *> neighbors = hnsw.FindNearest(query, 1);
  
  std::cout << "neighbors.size() is " << neighbors.size() << std::endl;
  std::cout << "neighbor vector was \n" << "{ " << neighbor.vec.Print() << " }" << std::endl;
  std::cout << "query vector was \n" << "{ " << query.vec.Print() << " }" << std::endl;

  if (neighbors.size())
    std::cout << "nearest vector was \n" << "{ " << neighbors[0]->vec.Print() << " }" << std::endl;
  else
    std::cout << "error -- we expected some output from the graph but got none." << std::endl;
}

void PrintMemUsage() {
  #ifdef _WIN32
    DWORD pid = GetCurrentProcessId();
    HANDLE handle = OpenProcess(PROCESS_VM_READ, false, pid);
    PROCESS_MEMORY_COUNTERS counters;
    GetProcessMemoryInfo(handle, &counters, counters.cb);
    CloseHandle(handle);
    std::cout << "PROCESS_MEMORY_COUNTERS: {" << std::endl;
    std::cout << "\tpage file usage (bytes): " << counters.PagefileUsage << std::endl;
    std::cout << "\tpeak page file usage (bytes): " << counters.PeakPagefileUsage << std::endl;
    std::cout << "}" << std::endl;
  #else
    
  #endif
}

void SizeTest(size_t count, size_t vector_length, size_t neighbors_per_node) {
  std::cout << "Sizeof a Vector is " << sizeof(struct Vector) << std::endl;
  std::cout << "Sizeof a std::vector<float> is " << sizeof(std::vector<float>) << std::endl;
  std::cout << "Sizeof a Node is " << sizeof(struct Hnsw::Node) << std::endl;

  std::cout << "allocating " << count << " vectors..." << std::endl;

  Vector **vec = new Vector*[count];
  for (size_t c = 0; c < count; ++c)
    vec[c] = new Vector(vector_length);

  PrintMemUsage();

  std::cout << "faulting-in " << count << " vectors..." << std::endl;

  for (size_t c = 0; c < count; ++c)
    for (size_t v = 0; v < vector_length; ++v)
      vec[c]->Set(v, v + c);

  PrintMemUsage();

  std::cout << "allocating " << count << " nodes, each of vector_length " << vector_length << "..." << std::endl;

  Hnsw::Node **nodes = new Hnsw::Node*[count];
  for (size_t c = 0; c < count; ++c)
    nodes[c] = new Hnsw::Node(vector_length);

  PrintMemUsage();

  std::cout << "faulting-in " << count << " nodes..." << std::endl;
  
  for (size_t c = 0; c < count; ++c)
    for (size_t v = 0; v < vector_length; ++v)
      nodes[c]->vec.Set(v, v + c);

  PrintMemUsage();

  std::cout << "faulting-in " << count << " node neighbors..." << std::endl;
  
  for (size_t c = 0; c < count; ++c)
    for (size_t n = 0; n < neighbors_per_node; ++n)
      nodes[c]->neighbors.push_back(nodes[n]);

  PrintMemUsage();
}

int main(int argc, char **argv) {
  SizeTest(256, 4, 2);
  return 0;
}