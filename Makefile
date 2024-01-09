unix:
	g++ -g -I /usr/local/include -L /usr/local/lib -o hnsw -std=c++17 hnsw.cc 
wind:
	g++ -g -I src/include -L src/lib -o hnsw hnsw.cc -std=c++17 -lmingw32 -lpsapi