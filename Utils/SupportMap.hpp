#ifndef __SUPPORTMAP_HPP__
#define __SUPPORTMAP_HPP__

#include <map>
#include <functional>
#include <iostream>

//c++ 17

namespace SupportMap {

	using std::map;
	using std::pair;


	template <typename Tk_, typename Tv_>
	pair<Tk_, Tv_> max(const map<Tk_, Tv_>& map_, function<bool(Tv_, Tv_)> comp = [](Tv_ a, Tv_ b) {return a > b; })
	{
		auto iter = map_.begin();
		Tv_ max_val = iter->second;
		Tk_ max_key = iter->first;

		for (++iter; iter != map_.end(); ++iter) {
			if (comp(iter->second, max_val)) {
				max_val = iter->second;
				max_key = iter->first;
			}
		}
		

		return { max_key, max_val };
	}

	template <typename Tk_, typename Tv_>
	std::ostream& operator<<(std::ostream& out, const map<Tk_, Tv_>& map_)
	{
		for (auto& [k, v] : map_) {
			std::cout << "class " << k << ": " << std::endl << v << std::endl;
		}

		return out;
	}
}

#endif