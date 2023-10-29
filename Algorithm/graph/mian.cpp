#include <iostream>
#include "graph.hpp"

using std::cout;
using std::endl;
using namespace DataStructure;

int main() {

	graph<int, EdgeListType::MATRIX> G;
	G.init("C:\\Users\\User\\Desktop\\C derived\\data\\graph input7.txt", true, true);
	function<void(int)> func = [](int v) -> void {cout << v << " "; };

	auto res = Bellman_Ford(G, 0);

	for (auto& [k, v] : res.dist) {
		cout << k << ": " << v << endl;
	}
}