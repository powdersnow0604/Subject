#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include <iostream>
#include <map>
#include <vector>
#include <stack>
#include <set>
#include <queue>
#include <array>
#include <fstream>
#include <string>
#include <algorithm>
#include <functional>
#include <limits>

using std::set;
using std::map;
using std::vector;
using std::stack;
using std::queue;
using std::priority_queue;
using std::array;
using std::string;
using std::cout;
using std::endl;
using std::function;

namespace DataStructure {

	//////////////////////////////////////////////////////////////////////////		class graph helpers		///////////////////////////////////////////////////////////////////////////////////////////////

	enum class EdgeListType
	{
		LIST,
		MATRIX
	};

	
	template <typename T>
	struct list_help_ {
		T vertex;
		double weight;
		list_help_(T v_, double w_ = 0): vertex(v_), weight(w_) {}
	};

	template <typename T>
	using list_help = list_help_<T>;

	typedef struct matrix_help_ {
		bool connected;
		double weight;
	}matrix_help;

	template <typename T>
	using ADJACENCYLIST = map<T, vector<list_help<T>>>;
	
	template <typename T>
	using ADJACENCYMATRIX = map<T, map<T, matrix_help>>;

	template <typename T, EdgeListType type>
	struct EDGELIST {
		ADJACENCYLIST<T> edge_list;

		vector<list_help<T>>& operator[](T vertex) { return edge_list[vertex];}
		const vector<list_help<T>>& operator[](T vertex) const { return edge_list.at(vertex); }
		ADJACENCYLIST<T>& operator()() { return edge_list; }
		const ADJACENCYLIST<T>& operator()() const { return edge_list; }
		EDGELIST<T, type>& operator=(const EDGELIST<T, type>& arg) { edge_list = arg(); return *this; }
	};

	template <typename T>
	struct EDGELIST<T, EdgeListType::MATRIX> {
		ADJACENCYMATRIX<T> edge_list;

		map<T, matrix_help>& operator[](T vertex) { return edge_list[vertex]; }
		const map<T, matrix_help>& operator[](T vertex) const { return edge_list.at(vertex); }
		ADJACENCYMATRIX<T>& operator()() { return edge_list; }
		const ADJACENCYMATRIX<T>& operator()() const { return edge_list; }
		EDGELIST<T, EdgeListType::MATRIX>& operator=(const EDGELIST<T, EdgeListType::MATRIX>& arg) { edge_list = arg(); return *this; }
	};

	//////////////////////////////////////////////////////////////////////////		class graph		///////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T, EdgeListType type = EdgeListType::LIST>
	class graph {
		size_t vertex_num;
		size_t edge_num;
		bool is_directed;
		bool is_weighted;
		struct EDGELIST<T,type> edge_list;

	public:
		void init(const char* path, bool isDirected = false, bool isWeighted = false);
		map<string, bool> getPropBool() const;
		map<string, size_t> getPropNum() const;
		void sort();
		void printEdge() const;
		struct EDGELIST<T, type>& getEdgeList();
		const struct EDGELIST<T, type>& getEdgeList() const;
		graph(): vertex_num(0), edge_num(0), is_directed(false), is_weighted(false) {}
		graph(size_t v_num, size_t e_num, bool is_d, bool is_w, const EDGELIST<T, type>& elist): 
			vertex_num(v_num), edge_num(e_num), is_directed(is_d), is_weighted(is_w), edge_list(elist) {}
	};


	//////////////////////////////////////////////////////////////////////////		class graph member function		///////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T, EdgeListType type>
	void graph<T, type>::init(const char* path, bool isDirected, bool isWeighted)
	{
		std::ifstream fin(path);
		set<T> verteces;
		size_t v_num, e_num;
		T from, to;
		double weight;

		is_directed = isDirected;
		is_weighted = isWeighted;

		fin >> v_num >> e_num;

		vertex_num = v_num;
		edge_num = e_num;

		
		for (size_t i = 0; i < e_num; ++i) {
			fin >> from >> to;

			verteces.insert(from);
			verteces.insert(to);

			if constexpr (type == EdgeListType::MATRIX) {
				if (is_weighted) {
					fin >> weight;
					edge_list[from][to].connected = true;
					edge_list[from][to].weight = weight;
					if (!is_directed) {
						edge_list[to][from].connected = true;
						edge_list[to][from].weight = weight;
					}
				}
				else {
					edge_list[from][to].connected = true;
					if (!is_directed) {
						edge_list[to][from].connected = true;
					}
				}

				for (auto& K : verteces) {
					edge_list[K];
				}

				for (auto& [K, V] : edge_list()) {
					for (auto& K : verteces) {
						V[K];
					}
				}
			}
			else {
				if (is_weighted) {
					fin >> weight;
					edge_list[from].push_back({ to, weight });
					if (!is_directed) edge_list[to].push_back({ from, weight });
				}
				else {
					edge_list[from].push_back({ to });
					if (!is_directed) edge_list[to].push_back({ from });
				}

				for (auto& K : verteces) {
					edge_list[K];
				}
			}
		}


		fin.close();
	}

	template <typename T, EdgeListType type>
	map<string, bool> graph<T, type>::getPropBool() const
	{
		return { {"is_directed", is_directed}, {"is_weighted", is_weighted} };
	}

	template <typename T, EdgeListType type>
	map<string, size_t> graph<T, type>::getPropNum() const
	{
		return { {"vertex_num", vertex_num}, {"edge_num", edge_num} };
	}

	template <typename T, EdgeListType type>
	void graph<T,type>::sort()
	{
		if constexpr (type == EdgeListType::MATRIX) return;
		else {
			for (auto& [V, L] : edge_list()) {
				std::sort(L.begin(), L.end());
			}
		}
	}

	template <typename T, EdgeListType type>
	void graph<T, type>::printEdge() const
	{
		if constexpr (type == EdgeListType::MATRIX){
			cout << "  ";
			for (auto& [K, V] : edge_list()) {
				cout << K << " ";
			}
			cout << endl;

			for (auto& [K, V] : edge_list()) {
				cout << K << " ";
				for (auto& [T, B] : V) {
					cout << B.connected << " ";
				}
				cout << endl;
			}
		}
		else {
			for (auto& [K, V] : edge_list()) {
				cout << K << ": ";
				for (auto& T : V) {
					cout << T.vertex << " ";
				}
				cout << endl;
			}
		}
	}

	template <typename T, EdgeListType type>
	struct EDGELIST<T, type>& graph<T, type>::getEdgeList() { return edge_list; }

	template <typename T, EdgeListType type>
	const struct EDGELIST<T, type>& graph<T, type>::getEdgeList() const { return edge_list; }

	//////////////////////////////////////////////////////////////////////////		dfs derived		///////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T, EdgeListType type>
	void dfsSubroutine(const EDGELIST<T,type>& edge_list, T vertex, map<T,bool>& visit, function<void(T)>& func, size_t& num, map<T, array<size_t, 2>>* pre_post = nullptr)
	{
		if (visit[vertex]) return;

		visit[vertex] = true;

		if(func != nullptr) func(vertex);

		if (pre_post != nullptr) (*pre_post)[vertex][0] = num++;

		if constexpr (type == EdgeListType::MATRIX) {
			for (auto& [K,V] : edge_list[vertex]) {
				if (!visit[K] && V.connected) dfsSubroutine(edge_list, K, visit, func, num, pre_post);
			}
		}
		else {
			for (auto& V : edge_list[vertex]) {
				if (!visit[V.vertex]) dfsSubroutine(edge_list, V.vertex, visit, func, num, pre_post);
			}
		}

		if (pre_post != nullptr) (*pre_post)[vertex][1] = num++;
	}

	template <typename T, EdgeListType type>
	void dfs(const graph<T, type>& G, const vector<T>& ud_order = vector<T>(), map<T, array<size_t, 2>>* pre_post = nullptr, function<void(T)> func = nullptr)
	{
		map<T, bool> visit;
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		size_t num = 1;

		for (auto& [K, V] : edge_list()) {
			visit[K] = false;
		}

		if (ud_order.size() == 0) {
			for (auto& [K, V] : edge_list()) {
				if (!visit[K]) dfsSubroutine(edge_list, K, visit, func, num, pre_post);
			}
		}
		else {
			for (auto& V : ud_order) {
				if (!visit[V]) dfsSubroutine(edge_list, V, visit, func, num, pre_post);
			}
		}
	}

	template <typename T, EdgeListType type>
	vector<vector<T>> CC(const graph<T, type>& G, const vector<T>& ud_order = vector<T>(), bool print = false)
	{
		map<T, bool> visit;
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		vector<vector<T>> ccs;
		vector<T> cc;
		size_t dummy;

		for (auto& [K, V] : edge_list()) {
			visit[K] = false;
		}

		function<void(T)> func = [&cc](T vertex) {cc.push_back(vertex); };

		if (ud_order.size() == 0) {
			for (auto& [K, V] : edge_list()) {
				if (!visit[K]) {
					dfsSubroutine(edge_list, K, visit, func, dummy);
					std::sort(cc.begin(), cc.end());
					ccs.push_back(cc);
					cc.clear();
				}
			}
		}
		else {
			for (auto& K : ud_order) {
				if (!visit[K]) {
					dfsSubroutine(edge_list, K, visit, func, dummy);
					std::sort(cc.begin(), cc.end());
					ccs.push_back(cc);
					cc.clear();
				}
			}
		}

		if (print) {
			cout << "connected components" << endl;
			for (auto& C : ccs) {
				for (auto& V : C) {
					cout << V << " ";
				}
				cout << endl;
			}
		}

		return ccs;
	}

	template <typename T, EdgeListType type>
	void topolgySubroutine(const EDGELIST<T, type>& edge_list, T vertex, map<T, bool>& visit, stack<T>& order, bool& is_sink)
	{
		if (visit[vertex]) return;

		visit[vertex] = true;

		size_t cnt = 0;

		if constexpr (type == EdgeListType::MATRIX) {
			for (auto& [K, V] : edge_list[vertex]) {
				if (!visit[K] && V.connected) topolgySubroutine(edge_list, K, visit, order, is_sink);
				++cnt;
				if (is_sink) break;
			}
		}
		else {
			for (auto& V : edge_list[vertex]) {
				if (!visit[V.vertex]) topolgySubroutine(edge_list, V.vertex, visit,  order, is_sink);
				++cnt;
				if (is_sink) break;
			}
		}

		order.push(vertex);

		if (cnt == 0) is_sink = true;
	}

	template <typename T, EdgeListType type>
	stack<T> topology(const graph<T, type>& G)
	{
		T source = T();
		size_t max = 0;
		map<T, bool> visit;
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		map<T, array<size_t, 2>> visit_order;
		stack<T> order;
		bool is_sink = false;

		function<void(T)> func = [](T arg) {};
		dfs(G, {}, &visit_order, func);

		for (auto& [K, V] : visit_order) {
			if (V[1] > max) {
				max = V[1];
				source = K;
			}
		}

		for (auto& [K, V] : edge_list()) {
			visit[K] = false;
		}

		topolgySubroutine(edge_list, source, visit, order, is_sink);

		return order;
	}

	template <typename T, EdgeListType type>
	EDGELIST<T, type> reverseGraph(const graph<T, type>& G) 
	{
		EDGELIST<T, type> reverse;
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		set<T> verteces;

		if constexpr (type == EdgeListType::MATRIX) {
			for (auto& [KF, VF] : edge_list()) {
				verteces.insert(KF);
				for (auto& [KT, VT] : VF) {
					reverse[KT][KF] = VT;
					verteces.insert(KT);
				}
			}

			for (auto& K : verteces) {
				reverse[K];
			}

			for (auto& [K, V] : reverse()) {
				for (auto& K : verteces) {
					V[K];
				}
			}
		}
		else {
			for (auto& [KF, VF] : edge_list()) {
				verteces.insert(KF);
				for (auto& VT : VF) {
					reverse[VT.vertex].push_back({ KF, (VT.weight) });
					verteces.insert(VT.vertex);
				}
			}

			for (auto& K : verteces) {
				reverse[K];
			}
		}

		return reverse;
	}

	template <typename T, EdgeListType type>
	vector<vector<T>> SCC(const graph<T, type>& G, bool print = false)
	{
		map<T, bool> visit;
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		size_t num = 1;
		map<T, array<size_t, 2>> visit_order;
		vector<T> ud_order;

		const EDGELIST<T, type>& reverseE = reverseGraph(G);
		auto Gpb = G.getPropBool();
		auto Gpn = G.getPropNum();

		for (auto& [K, V] : edge_list()) {
			visit[K] = false;
		}

		graph<T, type> reverseG = { Gpn["vertex_num"], Gpn["edge_num"], Gpb["is_directed"], Gpb["is_weighted"], reverseE };

		dfs(reverseG, {}, &visit_order);

		ud_order.reserve(Gpn["vertex_num"]);

		for (auto& [K, V] : visit_order) {
			auto i = ud_order.begin();
			for (; i != ud_order.end(); ++i) {
				if (visit_order[*i][1] < V[1]) break;
			}

			ud_order.insert(i, K);
		}

		return CC(G, ud_order, print);

	}

	template <typename T, EdgeListType type>
	void BCCSubroutine(const EDGELIST<T, type>& edge_list, T vertex, T parent, map<T,size_t>& dfn, map<T, size_t>& low, size_t& num, vector<set<T>>& bcc, stack<array<T,2>>& stack_ )
	{
		dfn[vertex] = low[vertex] = num++;

		if constexpr (type == EdgeListType::MATRIX) {
			for (auto& [K, V] : edge_list[vertex]) {
				if (V.connected && dfn[K] == 0) {
					stack_.push({ vertex, K });

					BCCSubroutine(edge_list, K, vertex, dfn, low, num, bcc, stack_);

					low[vertex] = low[vertex] > low[K] ? low[K] : low[vertex];

					if (low[K] >= dfn[vertex]) {
						bcc.push_back({});
						size_t i = bcc.size() - 1;
						array<T, 2> temp;
						do {
							temp = stack_.top();
							stack_.pop();
							bcc[i].insert(temp[0]);
							bcc[i].insert(temp[1]);
						} while (temp[0] != vertex && temp[1] != K);
					}
				}
				else if (V.connected && K != parent) {
					low[vertex] = low[vertex] > dfn[K] ? dfn[K] : low[vertex];
				}
			}
		}
		else {
			for (auto& V : edge_list[vertex]) {
				if (dfn[V.vertex] == 0) {
					stack_.push({ vertex, V.vertex });

					BCCSubroutine(edge_list, V.vertex, vertex, dfn, low, num, bcc, stack_);

					low[vertex] = low[vertex] > low[V.vertex] ? low[V.vertex] : low[vertex];

					if (low[V.vertex] >= dfn[vertex]) {
						bcc.push_back({});
						size_t i = bcc.size() - 1;
						array<T, 2> temp;
						do {
							temp = stack_.top();
							stack_.pop();
							bcc[i].insert(temp[0]);
							bcc[i].insert(temp[1]);
						} while (temp[0] != vertex && temp[1] != V.vertex);
					}
				}
				else if (V.vertex != parent) {
					low[vertex] = low[vertex] > dfn[V.vertex] ? dfn[V.vertex] : low[vertex];
				}
			}
		}
	}

	template <typename T, EdgeListType type>
	vector<set<T>> BCC(const graph<T, type> G, T root_parent = NULL, bool print = false)
	{
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		size_t num = 1;
		map<T, size_t> dfn;
		map<T, size_t> low;
		vector<set<T>> bcc;
		stack<array<T, 2>> stack_;

		for (auto& [K, V] : edge_list()) {
			dfn[K] = 0;
			low[K] = 0;
		}

		for (auto& [K, V] : edge_list()) {
			if (dfn[K] == 0) {
				BCCSubroutine(edge_list, K, root_parent, dfn, low, num, bcc, stack_);
				while (!stack_.empty()) stack_.pop();
			}
		}

		if (print) {
			cout << "biconnected component" << endl;
			for (auto& S : bcc) {
				for (auto& V : S) {
					cout << V << " ";
				}
				cout << endl;
			}
		}

		return bcc;
	}

	//////////////////////////////////////////////////////////////////////////		bfs derived		///////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T, EdgeListType type>
	void bfs(const graph<T, type>& G, T S, function<void(T)> func = nullptr)
	{
		map<T, size_t> dist;
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		queue<T> Q;
		T u;
		size_t dist_max = std::numeric_limits<size_t>::max();

		for (auto& [K, V] : edge_list()) {
			dist[K] = dist_max;
		}

		Q.push(S);
		dist[S] = 0;

		while (!Q.empty()) {
			u = Q.front();
			Q.pop();

			if (func != nullptr) func(u);
			

			if constexpr (type == EdgeListType::MATRIX) {
				for (auto& [K, V] : edge_list[u]) {
					if (dist[K] == dist_max && V.connected) {
						Q.push(K);
						dist[K] = dist[u] + 1;
					}
				}
			}
			else {
				for (auto& V : edge_list[u]) {
					if (dist[V.vertex] == dist_max) {
						Q.push(V.vertex);
						dist[V.vertex] = dist[u] + 1;
					}
				}
			}
		}
	}

	template <typename T>
	struct sssp {
		map<T, T> prev;
		map<T, double> dist;
		sssp(const map<T, T>& p, const map<T, double>& d) : prev(p), dist(d) {}
	};

	template <typename T, EdgeListType type>
	sssp<T> dijkstra(const graph<T, type>& G, T S, T null_value = NULL)
	{

		struct Qelement {
			T vertex;
			double distance;
			Qelement() : vertex(NULL), distance(0) {}
			Qelement(const T v, const double d) : vertex(v), distance(d) {}
			bool operator< (const Qelement& other) { return distance < other.distance; }
			bool operator== (const Qelement & other) { return vertex < other.vertex; }
			Qelement& operator= (const Qelement& other) { vertex = other.vertex; distance = other.distance; return *this; }
		};

		map<T, double> dist;
		map<T, T> prev;

		Qelement u;
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		vector<Qelement> Q;
		double dist_max = std::numeric_limits<double>::max();

		for (auto& [K, V] : edge_list()) {
			dist[K] = dist_max;
			prev[K] = null_value;
		}
		
		dist[S] = 0;

		Q.reserve(dist.size());

		for (auto& [K, V] : dist) {
			Q.push_back({ K, V });
		}


		while (!Q.empty()) {
			auto min_iter = std::min_element(Q.begin(), Q.end());
			u = *min_iter;
			Q.erase(min_iter);
			

			if constexpr (type == EdgeListType::MATRIX) {
				for (auto& [K, V] : edge_list[u.vertex]) {

					auto iter = std::find(Q.begin(), Q.end(), { K, 0 });
					std::find()
					if (V.connected && iter != Q.end() && dist[K] > dist[u.vertex] + V.wieght) {
						dist[K] = dist[u.vertex] + V.wieght;
						prev[K] = u.vertex;
						iter->distance = dist[K];
					}
				}
			}
			else {
				for (auto& V : edge_list[u]) {
					auto iter = std::find(Q.begin(), Q.end(), { V.vertex, 0 });
					if (iter != Q.end() && dist[V.vertex] > dist[u] + V.wieght) {
						dist[V.vertex] = dist[u] + V.wieght;
						prev[V.vertex] = u;
						iter->distance = dist[V.vertex];
					}
				}
			}

		}
		
		return { prev, dist };
	}

	template <typename T, EdgeListType type>
	sssp<T> Bellman_Ford(const graph<T, type>& G, T S, T null_value = NULL)
	{
		map<T, double> dist;
		map<T, T> prev;

		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		double dist_max = std::numeric_limits<double>::max();

		for (auto& [K, V] : edge_list()) {
			dist[K] = dist_max;
			prev[K] = null_value;
		}

		dist[S] = 0;

		for (auto& [K, V] : edge_list()) {

			if constexpr (type == EdgeListType::MATRIX) {
				for (auto& [T, W] : V) {
					if (W.connected && dist[T] > dist[K] + W.weight) {
						dist[T] = dist[K] + W.weight;
						prev[T] = K;
					}
				}
			}
			else {
				for (auto& W : V) {
					if (dist[W.vertex] > dist[K] + W.weight) {
						dist[W.vertex] = dist[K] + W.weight;
						prev[W.vertex] = K;
					}
				}
			}

		}

		return { prev, dist };
	}
	
	template <typename T, EdgeListType type>
	map<T, map<T, double>> Floyd(const graph<T, type>& G, bool print = false)
	{
		const EDGELIST<T, type>& edge_list = G.getEdgeList();
		double dist_max = std::numeric_limits<double>::max();
		map<T, map<T, double>> dist;
		set<T> verteces;
		

		if constexpr (type == EdgeListType::MATRIX) {
			for (auto& [KF, KV] : edge_list()) {
				verteces.insert(KF);
				for (auto& [KT, VT] : KV) {
					if (KF == KT) dist[KF][KT] = 0;
					else if (VT.connected) dist[KF][KT] = VT.weight;
					else dist[KF][KT] = dist_max;
				}
			
			}
		}
		else {
			for (auto& [KF, KV] : edge_list()) {
				verteces.insert(KF);
				for (auto& VT : KV) {
					verteces.insert(VT.vertex);
					dist[KF][VT.vertex] = VT.weight;
				}
			}

			for (auto& K : verteces) {
				dist[K];
			}

			for (auto& [KF, V] : dist) {
				for (auto& KT : verteces) {
					if (KF == KT) V[KT] = 0;
					if (V.find(KT) == V.end()) V[KT] = dist_max;
				}
			}
		}


		for (auto& K : verteces) {
			for (auto& I : verteces) {
				for (auto& J : verteces) {
					if (I == K || J == K || I == J) continue;
					if (dist[I][J] > dist[I][K] + dist[K][J])
						dist[I][J] = dist[I][K] + dist[K][J];
				}
			}
		}

		if (print) {
			cout << "  ";
			for (auto& [K, V] : dist) {
				cout << K << " ";
			}
			cout << endl;

			for (auto& [K, V] : dist) {
				cout << K << " ";
				for (auto& [T, B] : V) {
					cout << B << " ";
				}
				cout << endl;
			}
		}

		return dist;

	}
}
#endif
