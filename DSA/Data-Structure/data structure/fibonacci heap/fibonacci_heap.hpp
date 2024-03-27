#ifndef __FIBONACCI_HEAP_HPP__
#define __FIBONACCI_HEAP_HPP__

#include <functional>
#include <memory>
#include <cmath>
#include <vector>
#include <iostream>


namespace DataStructure {

	const double GOLDENRATIO = 1.61803398875;
	const double LOGGR = 0.481212; //std::log(GOLDENRATIO)

	template<typename T>
	struct fibo_node {
		fibo_node* parent;
		fibo_node* child;
		fibo_node* prev;
		fibo_node* next;
		T item;
		size_t degree;
		bool marked;

		void set_value(fibo_node* parent_, fibo_node* child_, fibo_node* prev_, fibo_node* next_, const T& item_, size_t degree_, bool marked_){
			child = child_;
			next = next_;
			prev = prev_;
			parent = parent_;
			degree = degree_;
			item = item_;
			marked = marked_;
		}
	};
	
	template<typename T, typename Compare = std::less<T>, typename Allocator = std::allocator<fibo_node<T>>>
	class fibonacci_heap {
	public:
		typedef fibo_node<T> node;
		typedef T value_type;

	private:
		node* min;
		Compare comp;
		Allocator allocator;
		size_t node_num = 0;

		void consolidate();
		void link_2_node(node* n1, node* n2);
		void add_node(node* dest, node* source);
		void prunning(node* node_);
		void deleter(node* node_);

	public:
		fibonacci_heap(const Compare& comp_ = Compare(), const Allocator& allocator_ = Allocator() ) : min(nullptr), comp(comp_), allocator(allocator_) {}
		node* insert(const T& item);
		T get_min() const;
		T extract_min();
		void decrease_key(node* key, const T& value);
		void print();
		bool is_empty() { return min == nullptr; }
		~fibonacci_heap() noexcept;
	};

	//////////////////////////////////////////////////////////////////////////////////////////////////////////////
	template<typename T, typename Compare, typename Allocator>
	void fibonacci_heap<T, Compare, Allocator>::print()
	{
		using std::cout;
		using std::endl;

		if (min == nullptr) return;

		node* curr = min;
		do {
			cout << "[" << curr->item << "]" << endl;
			cout << "next: " << curr->next->item << endl;
			cout << "prev: " << curr->prev->item << endl;
			cout << "degree: " << curr->degree << endl << endl;

			curr = curr->next;
		} while (curr != min);
	}

	template<typename T, typename Compare, typename Allocator>
	fibo_node<T>* fibonacci_heap<T, Compare, Allocator>::insert(const T& item)
	{
		++node_num;

		if (min == nullptr) {
			min = allocator.allocate(1);
			min->set_value(nullptr, nullptr, min, min, item, 0, false);

			return min;
		}

		node* nnode = allocator.allocate(1);
		nnode->set_value(nullptr, nullptr, min->prev, min, item, 0, false);

		min->prev->next = nnode;
		min->prev = nnode;

		if (comp(item, min->item)) {
			min = nnode;
		}

		return nnode;
	}
	
	template<typename T, typename Compare, typename Allocator>
	T fibonacci_heap<T, Compare, Allocator>::get_min() const
	{
		return min->item;
	}

	template<typename T, typename Compare, typename Allocator>
	T fibonacci_heap<T, Compare, Allocator>::extract_min()
	{
		if (min == nullptr) return T();

		if (node_num == 1) {
			node_num = 0;
			T temp = min->item;
			allocator.deallocate(min, 1);
			min = nullptr;
			return temp;
		}


		node* prev_min = min;
		T min_value = min->item;

		if (min->child != nullptr) {
			link_2_node(min, min->child);
			min->child = nullptr;
		}
		
		prev_min->prev->next = prev_min->next;
		prev_min->next->prev = prev_min->prev;

		min = prev_min->next;

		allocator.deallocate(prev_min, 1);
		
		--node_num;
		consolidate();

		return min_value;
	}

	template<typename T, typename Compare, typename Allocator>
	void fibonacci_heap<T, Compare, Allocator>::consolidate()
	{
		size_t degree, max_degree = static_cast<size_t>(std::log(node_num) / LOGGR);
		node* curr = min, * bigger, * smaller, * candidate, * last = min->prev;
		bool running = true;
		std::vector<node*> vec(max_degree + 1, nullptr);

		do {
			if (comp(curr->item, min->item)) {
				min = curr;
			}

			if (curr == last) running = false;

			curr->parent = nullptr;
			degree = curr->degree;
			candidate = curr;
			curr = curr->next;

			while (degree <= max_degree) {
				if (vec[degree] == nullptr) {
					vec[degree] = candidate;
					break;
				}

				if (comp(candidate->item, vec[degree]->item)) {
					smaller = candidate;
					bigger = vec[degree];
				}
				else {
					smaller = vec[degree];
					bigger = candidate;
				}

				if (smaller->child == nullptr) {
					bigger->next->prev = bigger->prev;
					bigger->prev->next = bigger->next;

					smaller->child = bigger;
					bigger->parent = smaller;
					bigger->next = bigger->prev = bigger;
				}
				else {
					bigger->next->prev = bigger->prev;
					bigger->prev->next = bigger->next;

					bigger->parent = smaller;

					add_node(smaller->child, bigger);
				}

				vec[degree] = nullptr;
				++(smaller->degree);
				++degree;
				candidate = smaller;
			}
		} while (running);
	}

	template<typename T, typename Compare, typename Allocator>
	void fibonacci_heap<T, Compare, Allocator>::decrease_key(fibonacci_heap<T, Compare, Allocator>::node* key, const T& value)
	{
		if (comp(key->item, value)) return;

		key->item = value;
		
		if (key->parent != nullptr) {

			if (key->next != key) {
				key->next->prev = key->prev;
				key->prev->next = key->next;
				if (key->parent->child == key) {
					key->parent->child = key->next;
				}
			}
			else {
				key->parent->child = nullptr;
			}

			add_node(min, key);

			if (comp(key->item, min->item)) {
				min = key;
			}

			prunning(key->parent);
			key->parent = nullptr;
		}
		else {
			if (comp(key->item, min->item)) {
				min = key;
			}
		}
	}

	template<typename T, typename Compare, typename Allocator>
	void fibonacci_heap<T, Compare, Allocator>::prunning(node* node_)
	{
		if (node_->marked) {
			node_->marked = false;
			
			if (node_->parent != nullptr) {

				if (node_->next != node_) {
					node_->next->prev = node_->prev;
					node_->prev->next = node_->prev;
				}

				add_node(min, node_);

				if (comp(node_->item, min->item)) {
					min = node_;
				}

				prunning(node_->parent);
				node_->parent = nullptr;
			}
			else {
				if (comp(node_->item, min->item)) {
					min = node_;
				}
			}
		}
		else {
			node_->marked = true;
		}
	}

	template<typename T, typename Compare, typename Allocator>
	void fibonacci_heap<T, Compare, Allocator>::link_2_node(node* n1, node* n2)
	{
		if (n1 == nullptr) return;
		if (n2 == nullptr) return;

		node* temp = n1->prev;
		n1->prev->next = n2;
		n2->prev->next = n1;
		n1->prev = n2->prev;
		n2->prev = temp;
	}

	template<typename T, typename Compare, typename Allocator>
	void fibonacci_heap<T, Compare, Allocator>::add_node(node* dest, node* source)
	{
		if (dest == nullptr) return;
		if (source == nullptr) return;

		source->next = dest;
		source->prev = dest->prev;
		dest->prev->next = source;
		dest->prev = source;

	}

	template<typename T, typename Compare, typename Allocator>
	fibonacci_heap<T, Compare, Allocator>::~fibonacci_heap() noexcept
	{
		if (min != nullptr) {
			node* curr = min;
			node* temp;
			min->prev->next = nullptr;
			do {
				if (curr->child != nullptr) {
					deleter(curr->child);
				}
				temp = curr;
				curr = curr->next;
				allocator.deallocate(temp, 1);
			} while (curr != nullptr);
		}
	}
	
	template<typename T, typename Compare, typename Allocator>
	void fibonacci_heap<T, Compare, Allocator>::deleter(node* node_)
	{
		node* curr = node_;
		node* temp;
		node_->prev->next = nullptr;
		do {
			if (curr->child != nullptr) {
				deleter(curr->child);
			}
			temp = curr;
			curr = curr->next;
			allocator.deallocate(temp, 1);
		} while (curr != nullptr);
	}
}

#endif
