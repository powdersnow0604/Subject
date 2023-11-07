#ifndef __GRAPH_HPP__
#define __GRAPH_HPP__

#include <iostream>
#include <map>
#include <vector>
#include <stack>
#include <forward_list>
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
using std::forward_list;
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
	struct edge {
		T from;
		T to;
		double weight;
		edge(T f, T t, double w): from(f), to(t), weight(w) {}
	};
	
	typedef struct list_help_ {
		size_t vertex;
		double weight;
		list_help_(size_t v_, double w_ = 0): vertex(v_), weight(w_) {}
		bool operator< (const list_help_& other) const { return vertex < other.vertex; }
	}list_help;

	typedef struct matrix_help_ {
		bool connected;
		double weight;
	}matrix_help;

	using ADJACENCYLIST = vector<forward_list<list_help>>;
	
	using ADJACENCYMATRIX = vector<vector<matrix_help>>;

	template <EdgeListType type>
	struct EDGELIST {
		ADJACENCYLIST edge_list;

		forward_list<list_help>& operator[](size_t vertex) { return edge_list[vertex];}
		const forward_list<list_help>& operator[](size_t vertex) const { return edge_list[vertex]; }
		ADJACENCYLIST& operator()() { return edge_list; }
		const ADJACENCYLIST& operator()() const { return edge_list; }
		EDGELIST<type>& operator=(const EDGELIST<type>& arg) { edge_list = arg(); return *this; }
	};

	template <>
	struct EDGELIST<EdgeListType::MATRIX> {
		ADJACENCYMATRIX edge_list;

		vector<matrix_help>& operator[](size_t vertex) { return edge_list[vertex]; }
		const vector<matrix_help>& operator[](size_t vertex) const { return edge_list[vertex]; }
		ADJACENCYMATRIX& operator()() { return edge_list; }
		const ADJACENCYMATRIX& operator()() const { return edge_list; }
		EDGELIST<EdgeListType::MATRIX>& operator=(const EDGELIST<EdgeListType::MATRIX>& arg) { edge_list = arg(); return *this; }
	};

	//////////////////////////////////////////////////////////////////////////		class graph		///////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T, EdgeListType type = EdgeListType::LIST>
	class graph {
		map<size_t, T> class_table;
		size_t vertex_num;
		size_t edge_num;
		bool is_directed;
		bool is_weighted;
		struct EDGELIST<type> edge_list;

	public:
		void init(const char* path, bool isDirected = false, bool isWeighted = false);
		map<string, bool> getPropBool() const;
		map<string, size_t> getPropNum() const;
		map<size_t, T> getClassTable() const;
		void sort();
		void printEdge() const;
		struct EDGELIST<type>& getEdgeList();
		const struct EDGELIST<type>& getEdgeList() const;
		graph(): vertex_num(0), edge_num(0), is_directed(false), is_weighted(false) {}
		graph(size_t v_num, size_t e_num, bool is_d, bool is_w, const EDGELIST<type>& elist, const map<size_t, T> ct): 
			vertex_num(v_num), edge_num(e_num), is_directed(is_d), is_weighted(is_w), edge_list(elist), class_table(ct) {}
	};


	//////////////////////////////////////////////////////////////////////////		class graph member function		///////////////////////////////////////////////////////////////////////////////////////////////

	template <typename T, EdgeListType type>
	void graph<T, type>::init(const char* path, bool isDirected, bool isWeighted)
	{
		std::ifstream fin(path);
		map<T, size_t> class_map;
		T from, to;
		size_t num = 0, F, T;
		double weight;

		is_directed = isDirected;
		is_weighted = isWeighted;

		fin >> vertex_num >> edge_num;


		if constexpr (type == EdgeListType::MATRIX) {
			edge_list().resize(vertex_num);
			for (auto& v : edge_list()) {
				v.resize(vertex_num, {false, 0});
			}

			for (size_t i = 0; i < edge_num; ++i) {
				fin >> from >> to;

				if (class_map.find(from) == class_map.end()) {
					class_map[from] = num++;
				}
				if (class_map.find(to) == class_map.end()) {
					class_map[to] = num++;
				}

				F = class_map[from];
				T = class_map[to];

				if (is_weighted) {
					fin >> weight;
					edge_list[F][T].connected = true;
					edge_list[F][T].weight = weight;
					if (!is_directed) {
						edge_list[T][F].connected = true;
						edge_list[T][F].weight = weight;
					}
				}
				else {
					edge_list[F][T].connected = true;
					if (!is_directed) {
						edge_list[T][F].connected = true;
					}
				}
			}
		}
		else {
			edge_list().resize(vertex_num);

			for (size_t i = 0; i < edge_num; ++i) {
				fin >> from >> to;

				if (class_map.find(from) == class_map.end()) {
					class_map[from] = num++;
				}
				if (class_map.find(to) == class_map.end()) {
					class_map[to] = num++;
				}

				F = class_map[from];
				T = class_map[to];

				if (is_weighted) {
					fin >> weight;
					edge_list[F].push_front({ T, weight });
					if (!is_directed) edge_list[T].push_front({ F, weight });
				}
				else {
					edge_list[F].push_front({ T, 0 });
					if (!is_directed) edge_list[T].push_front({ F, 0 });
				}
			}
		}

		for (auto& [k, v] : class_map) {
			class_table[v] = k;
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

	template<typename T, EdgeListType type>
	map<size_t, T> graph<T,type>::getClassTable() const {
		return class_table;
	}


	template <typename T, EdgeListType type>
	void graph<T,type>::sort()
	{
		if constexpr (type == EdgeListType::MATRIX) return;
		else {
			priority_queue<list_help> Q;
			for (size_t i = 0; i < vertex_num; ++i) {
				while (!edge_list[i].empty()) {
					Q.push(edge_list[i].front());
					edge_list[i].pop_front();
				}

				while (!Q.empty()) {
					edge_list[i].push_front(Q.top());
					Q.pop();
				}
			}
		}
	}

	template <typename T, EdgeListType type>
	void graph<T, type>::printEdge() const
	{
		if constexpr (type == EdgeListType::MATRIX){
			cout << "  ";
			for (size_t i = 0; i < vertex_num; ++i) {
				cout << class_table.at(i) << " ";
			}
			cout << endl;

			for (size_t i = 0; i < vertex_num; ++i) {
				cout << class_table.at(i) << " ";
				for (size_t j = 0; j < vertex_num; ++j) {
					cout << edge_list[i][j].connected << " ";
				}
				cout << endl;
			}
		}
		else {
			for (size_t i = 0; i < vertex_num; ++i) {
				cout << class_table.at(i) << ": ";
				auto iter = edge_list[i].begin();
				while (iter != edge_list[i].end()) {
					cout << class_table.at((*iter).vertex) << " ";
					++iter;
				}
				cout << endl;
			}

		}
	}

	template <typename T, EdgeListType type>
	struct EDGELIST<type>& graph<T, type>::getEdgeList() { return edge_list; }

	template <typename T, EdgeListType type>
	const struct EDGELIST<type>& graph<T, type>::getEdgeList() const { return edge_list; }

	//////////////////////////////////////////////////////////////////////////		dfs derived		///////////////////////////////////////////////////////////////////////////////////////////////
	
	template <EdgeListType type>
	void dfsSubroutine(const EDGELIST<type>& edge_list, size_t vertex, vector<bool>& visit, function<void(size_t)>& func, size_t& num, vector<array<size_t, 2>>* pre_post = nullptr)
	{
		if (visit[vertex]) return;

		visit[vertex] = true;

		if(func != nullptr) func(vertex);

		if (pre_post != nullptr) (*pre_post)[vertex][0] = num++;

		if constexpr (type == EdgeListType::MATRIX) {
			for (size_t i = 0; i < edge_list[vertex].size(); ++i) {
				if (!visit[i] && edge_list[vertex][i].connected) dfsSubroutine(edge_list, i, visit, func, num, pre_post);
			}
		}
		else {
			auto iter = edge_list[vertex].begin();
			
			while (iter != edge_list[vertex].end()) {
				if (!visit[(*iter).vertex]) dfsSubroutine(edge_list, (*iter).vertex, visit, func, num, pre_post);
				++iter;
			}
		}

		if (pre_post != nullptr) (*pre_post)[vertex][1] = num++;
	}

	template <typename T, EdgeListType type>
	void dfs(const graph<T, type>& G, const vector<size_t>& ud_order = vector<size_t>(), vector<array<size_t, 2>>* pre_post = nullptr, function<void(size_t)> func = nullptr)
	{
		vector<bool> visit;
		const EDGELIST<type>& edge_list = G.getEdgeList();
		size_t num = 1;
		size_t v_num = edge_list().size();
		visit.resize(v_num, false);

		if (ud_order.size() == 0) {
			for (size_t i = 0; i < v_num; ++i) {
				if (!visit[i]) dfsSubroutine(edge_list, i, visit, func, num, pre_post);
			}
		}
		else {
			for (auto& V : ud_order) {
				if (!visit[V]) dfsSubroutine(edge_list, V, visit, func, num, pre_post);
			}
		}
	}
	
	template <typename T, EdgeListType type>
	vector<vector<size_t>> CC(const graph<T, type>& G, const vector<size_t>& ud_order = vector<size_t>(), bool print = false)
	{
		vector<bool> visit;
		const EDGELIST<type>& edge_list = G.getEdgeList();
		vector<vector<size_t>> ccs;
		vector<size_t> cc;
		size_t dummy;
		map<size_t, T> class_table = G.getClassTable();

		size_t v_num = edge_list().size();
		visit.resize(v_num, false);

		function<void(size_t)> func = [&cc](size_t vertex) {cc.push_back(vertex); };

		if (ud_order.size() == 0) {
			for (size_t i = 0; i < v_num; ++i) {
				if (!visit[i]) {
					dfsSubroutine(edge_list, i, visit, func, dummy);
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
					cout << class_table[V] << " ";
				}
				cout << endl;
			}
		}

		return ccs;
	}
	
	template <EdgeListType type>
	void topolgySubroutine(const EDGELIST<type>& edge_list, size_t vertex, vector<bool>& visit, stack<size_t>& order, bool& is_sink)
	{
		if (visit[vertex]) return;

		visit[vertex] = true;

		size_t cnt = 0;

		if constexpr (type == EdgeListType::MATRIX) {
			for (size_t i = 0; i < edge_list[vertex].size(); ++i) {
				if (!visit[i] && edge_list[vertex][i].connected) {
					topolgySubroutine(edge_list, i, visit, order, is_sink);
					++cnt;
				}
				if (is_sink) break;
			}
		}
		else {
			auto iter = edge_list[vertex].begin();

			while (iter != edge_list[vertex].end()) {
				if (!visit[(*iter).vertex]) topolgySubroutine(edge_list, (*iter).vertex, visit, order, is_sink);
				++iter;
				++cnt;
				if (is_sink) break;
			}
		}

		order.push(vertex);

		if (cnt == 0) is_sink = true;
	}

	template <typename T, EdgeListType type>
	stack<size_t> topology(const graph<T, type>& G)
	{
		size_t source = 0;
		size_t max = 0;
		vector<bool> visit;
		const EDGELIST<type>& edge_list = G.getEdgeList();
		vector<array<size_t, 2>> visit_order;
		stack<size_t> order;
		bool is_sink = false;
		size_t v_num = edge_list().size();

		visit_order.resize(v_num);
		function<void(size_t)> func = [](size_t arg) {};
		dfs(G, {}, &visit_order, func);

		for (size_t i = 0; i < v_num; ++i) {
			if (visit_order[i][1] > max) {
				max = visit_order[i][1];
				source = i;
			}
		}

		visit.resize(v_num, false);

		topolgySubroutine(edge_list, source, visit, order, is_sink);

		return order;
	}
	
	template <typename T, EdgeListType type>
	EDGELIST<type> reverseGraph(const graph<T, type>& G) 
	{
		EDGELIST<type> reverse;
		const EDGELIST<type>& edge_list = G.getEdgeList();
		size_t v_num = edge_list().size();

		if constexpr (type == EdgeListType::MATRIX) {

			reverse().resize(v_num);
			for (auto& v : reverse()) {
				v.resize(v_num);
			}

			for (size_t i = 0; i < v_num; ++i) {
				for (size_t j = 0; j < v_num; ++j) {
					reverse[j][i] = edge_list[i][j];
				}
			}
		}
		else {

			reverse().resize(v_num);

			for (size_t i = 0; i < v_num; ++i) {
				auto iter = edge_list[i].begin();
				while (iter != edge_list[i].end()) {
					reverse[iter->vertex].push_front({ i, iter->weight });
					++iter;
				}
			}
		}

		return reverse;
	}

	template <typename T, EdgeListType type>
	vector<vector<size_t>> SCC(const graph<T, type>& G, bool print = false)
	{
		vector<bool> visit;
		const EDGELIST<type>& edge_list = G.getEdgeList();
		size_t num = 1;
		vector<array<size_t, 2>> visit_order;
		vector<size_t> ud_order;
		size_t v_num = edge_list().size();

		const EDGELIST<type>& reverseE = reverseGraph(G);
		auto Gpb = G.getPropBool();
		auto Gpn = G.getPropNum();
		auto ct = G.getClassTable();

		visit.resize(v_num, false);
		visit_order.resize(v_num);

		graph<T, type> reverseG = { Gpn["vertex_num"], Gpn["edge_num"], Gpb["is_directed"], Gpb["is_weighted"], reverseE, ct };

		dfs(reverseG, {}, &visit_order);

		ud_order.reserve(Gpn["vertex_num"]);

		for (size_t k = 0; k < v_num; ++k) {
			auto i = ud_order.begin();
			for (; i != ud_order.end(); ++i) {
				if (visit_order[*i][1] < visit_order[k][1]) break;
			}

			ud_order.insert(i, k);
		}

		

		return CC(G, ud_order, print);

	}
	
	template <EdgeListType type>
	void BCCSubroutine(const EDGELIST<type>& edge_list, size_t vertex, size_t parent, vector<size_t>& dfn, vector<size_t>& low, size_t& num, vector<set<size_t>>& bcc, stack<array<size_t,2>>& stack_ )
	{
		dfn[vertex] = low[vertex] = num++;
		

		if constexpr (type == EdgeListType::MATRIX) {
			for (size_t i = 0; i < edge_list[vertex].size(); ++i) {
				if (edge_list[vertex][i].connected && dfn[i] == 0) {
					stack_.push({ vertex, i });

					BCCSubroutine(edge_list, i, vertex, dfn, low, num, bcc, stack_);

					low[vertex] = low[vertex] > low[i] ? low[i] : low[vertex];

					if (low[i] >= dfn[vertex]) {
						bcc.push_back({});
						size_t j = bcc.size() - 1;
						array<size_t, 2> temp;
						do {
							temp = stack_.top();
							stack_.pop();
							bcc[j].insert(temp[0]);
							bcc[j].insert(temp[1]);

						} while (temp[0] != vertex && temp[1] != i);
					}
				}
				else if (edge_list[vertex][i].connected && i != parent) {
					low[vertex] = low[vertex] > dfn[i] ? dfn[i] : low[vertex];
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
						array<size_t, 2> temp;
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
	vector<set<size_t>> BCC(const graph<T, type> G, T root_parent = NULL, bool print = false)
	{
		const EDGELIST<type>& edge_list = G.getEdgeList();
		size_t num = 1;
		vector<size_t> dfn;
		vector<size_t> low;
		vector<set<size_t>> bcc;
		stack<array<size_t, 2>> stack_;
		size_t v_num = edge_list().size();

		dfn.resize(v_num);
		low.resize(v_num);

		for (size_t i = 0; i < v_num; ++i) {
			dfn[i] = 0;
			low[i] = 0;
		}

		for (size_t i = 0; i < v_num; ++i) {
			if (dfn[i] == 0) {
				BCCSubroutine(edge_list, i, root_parent, dfn, low, num, bcc, stack_);
				while (!stack_.empty()) stack_.pop();
			}
		}

		


		if (print) {
			auto ct = G.getClassTable();
			cout << "biconnected component" << endl;
			for (auto& S : bcc) {
				for (auto& V : S) {
					cout << ct[V] << " ";
				}
				cout << endl;
			}
		}

		return bcc;
	}

	//////////////////////////////////////////////////////////////////////////		bfs derived		///////////////////////////////////////////////////////////////////////////////////////////////
	
	template <typename T, EdgeListType type>
	void bfs(const graph<T, type>& G, size_t S, function<void(size_t)> func = nullptr)
	{
		vector<size_t> dist;
		const EDGELIST<type>& edge_list = G.getEdgeList();
		queue<size_t> Q;
		size_t u;
		constexpr size_t dist_max = std::numeric_limits<size_t>::max();
		size_t v_num = edge_list().size();

		dist.resize(v_num, dist_max);

		Q.push(S);
		dist[S] = 0;

		while (!Q.empty()) {
			u = Q.front();
			Q.pop();

			if (func != nullptr) func(u);
			

			if constexpr (type == EdgeListType::MATRIX) {
				for (size_t i = 0; i < v_num; ++i) {
					if (edge_list[u][i].connected && dist[i] == dist_max) {
						Q.push(i);
						dist[i] = dist[u] + 1;
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
	
	struct sssp {
		vector<size_t> prev;
		vector<double> dist;
		sssp(const vector<size_t>& p, const vector<double>& d) : prev(p), dist(d) {}
	};

	template <typename T, EdgeListType type>
	sssp dijkstra(const graph<T, type>& G, size_t S, size_t null_value = NULL)
	{
		vector<double> dist;
		vector<size_t> prev;

		size_t u;
		const EDGELIST<type>& edge_list = G.getEdgeList();
		vector<size_t> Q;
		constexpr double dist_max = std::numeric_limits<double>::max();
		size_t v_num = edge_list().size();

		dist.resize(v_num, dist_max);
		prev.resize(v_num, null_value);
		
		dist[S] = 0;

		Q.reserve(v_num);

		for (size_t i = 0; i < v_num; ++i) {
			Q.push_back(i);
		}

		while (!Q.empty()) {

			auto min_iter = std::min_element(Q.begin(), Q.end(), [&dist](const size_t& arg1, const size_t& arg2) {return dist[arg1] < dist[arg2]; });
			u = *min_iter;
			Q.erase(min_iter);

			if constexpr (type == EdgeListType::MATRIX) {
				for (size_t i = 0; i < v_num; ++i) {

					auto iter = std::find(Q.begin(), Q.end(), i);
					if (edge_list[u][i].connected && iter != Q.end() && dist[i] > dist[u] + edge_list[u][i].weight) {
						dist[i] = dist[u] + edge_list[u][i].weight;
						prev[i] = u;
					}
				}
			}
			else {
				for (auto& V : edge_list[u]) {
					auto iter = std::find(Q.begin(), Q.end(), V.vertex);
					if (iter != Q.end() && dist[V.vertex] > dist[u] + V.weight) {
						dist[V.vertex] = dist[u] + V.weight;
						prev[V.vertex] = u;
					}
				}
			}

		}
		
		return { prev, dist };
	}
	
	template <typename T, EdgeListType type>
	sssp Bellman_Ford(const graph<T, type>& G, size_t S, size_t null_value = NULL)
	{
		vector<double> dist;
		vector<size_t> prev;

		const EDGELIST<type>& edge_list = G.getEdgeList();
		constexpr double dist_max = std::numeric_limits<double>::max();
		size_t v_num = edge_list().size();

		dist.resize(v_num, dist_max);
		prev.resize(v_num, null_value);

		dist[S] = 0;

		for (size_t k = 0; k < v_num; ++k) {

			if constexpr (type == EdgeListType::MATRIX) {
				for (size_t i = 0; i < edge_list[k].size(); ++i) {
					if (edge_list[k][i].connected && dist[i] > dist[k] + edge_list[k][i].weight) {
						dist[i] = dist[k] + edge_list[k][i].weight;
						prev[i] = k;
					}
				}
			}
			else {
				for (auto& W : edge_list[k]) {
					if (dist[W.vertex] > dist[k] + W.weight) {
						dist[W.vertex] = dist[k] + W.weight;
						prev[W.vertex] = k;
					}
				}
			}

		}

		return { prev, dist };
	}
	
	template <typename T, EdgeListType type>
	vector<vector<double>> Floyd(const graph<T, type>& G, bool print = false)
	{
		const EDGELIST<type>& edge_list = G.getEdgeList();
		constexpr double dist_max = std::numeric_limits<double>::max();
		vector<vector<double>> dist;
		size_t v_num = edge_list().size();

		dist.resize(v_num,vector<double>(v_num, dist_max));
		

		if constexpr (type == EdgeListType::MATRIX) {
			for (size_t i = 0; i < v_num; ++i) {
				for (size_t j = 0; j < v_num; ++j) {
					if (i == j) {
						dist[i][j] = 0;
						continue;
					}
					if (!edge_list[i][j].connected) {
						continue;
					}
					dist[i][j] = edge_list[i][j].weight;
				}
			}
		}
		else {
			for (size_t i = 0; i < v_num; ++i) {
				dist[i][i] = 0;
				for (auto& v : edge_list[i]) {
					dist[i][v.vertex] = v.weight;
				}
			}
		}


		for (size_t k = 0; k < v_num; ++k) {
			for (size_t i = 0; i < v_num; ++i) {
				for (size_t j = 0; j < v_num; ++j) {
					if (i == k || j == k || i == j) continue;
					if (dist[i][j] > dist[i][k] + dist[k][j])
						dist[i][j] = dist[i][k] + dist[k][j];
				}
			}
		}
		
		if (print) {
			auto class_table = G.getClassTable();
			cout << "  ";
			for (size_t i = 0; i < v_num; ++i) {
				cout << class_table.at(i) << " ";
			}
			cout << endl;

			for (size_t i = 0; i < v_num; ++i) {
				cout << class_table.at(i) << " ";
				for (size_t j = 0; j < v_num; ++j) {
					cout << dist[i][j] << " ";
				}
				cout << endl;
			}
		}

		return dist;

	}

	struct uf_set {
		size_t root;
		vector<size_t> child;
		uf_set(size_t r) : root(r) {}
	};

	void uf_union_helper(vector<uf_set>& set, size_t nroot, size_t target)
	{
		uf_set& tset = set[target];
		tset.root = nroot;
		for (size_t i = 0; i < tset.child.size(); ++i) {
			uf_union_helper(set, nroot, tset.child[i]);
		}
	}

	template <typename T, EdgeListType type>
	vector<edge<size_t>> Kruskal(const graph<T, type>& G)
	{
		const EDGELIST<type>& edge_list = G.getEdgeList();
		size_t v_num = edge_list().size();
		size_t e_num = G.getPropNum()["edge_num"];
		bool is_directed = G.getPropBool()["is_directed"];
		size_t i;
		
		vector<edge<size_t>> edges;
		vector<edge<size_t>> mst;
		vector<uf_set> eset;

		edges.reserve(e_num);
		mst.reserve(v_num - 1);
		eset.reserve(v_num);

		if constexpr (type == EdgeListType::MATRIX) {
			if (is_directed) {
				for (i = 0; i < v_num; ++i) {
					for (size_t j = 0; j < v_num; ++j) {
						if (edge_list[i][j].connected) {
							edges.push_back({ i,j,edge_list[i][j].weight });
						}
					}
				}
			}
			else {
				for (i = 0; i < v_num; ++i) {
					for (size_t j = i; j < v_num; ++j) {
						if (edge_list[i][j].connected) {
							edges.push_back({ i,j,edge_list[i][j].weight });
						}
					}
				}
			}
		}
		else {
			if (is_directed) {
				for (i = 0; i < v_num; ++i) {
					for (auto& E : edge_list[i]) {
						edges.push_back({ i,E.vertex,E.weight });
					}
				}
			}
			else {
				for (i = 0; i < v_num; ++i) {
					for (auto& E : edge_list[i]) {
						if (i > E.vertex) continue;
						edges.push_back({ i,E.vertex,E.weight });
					}
				}
			}
		}

		std::sort(edges.begin(), edges.end(), [](const edge<size_t>& E1, const edge<size_t>& E2) {return E1.weight < E2.weight; });

		for (i = 0; i < v_num; ++i) {
			eset.push_back({ i });
		}

		for (i = 0; i < e_num; ++i) {
			if (eset[edges[i].from].root != eset[edges[i].to].root) {
				mst.push_back(edges[i]);
				if (eset[edges[i].from].root > eset[edges[i].to].root) {
					eset[edges[i].to].child.push_back(edges[i].from);
					uf_union_helper(eset, eset[edges[i].to].root, eset[edges[i].from].root);
				}
				else {
					eset[edges[i].from].child.push_back(edges[i].to);
					uf_union_helper(eset, eset[edges[i].from].root, eset[edges[i].to].root);
				}


			}
		}

		if (mst.size() == 0) return {};

		return mst;
	}
	
}
#endif
