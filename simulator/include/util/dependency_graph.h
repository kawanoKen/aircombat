// Copyright (c) 2024-2025 Air Systems Research Center, Acquisition, Technology & Logistics Agency(ATLA)
//処理順序を管理するためのDAGの一般化
#pragma once
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/topological_sort.hpp>
#include <boost/heap/fibonacci_heap.hpp>
#include <cereal/cereal.hpp>
#include <cereal/details/util.hpp>
#include <deque>
#include <algorithm>
#include <ranges>
#include "macros/common_macros.h"
#include "../py_init_hook.h"
#include "../traits/converter.h"
#include "range_iterable.h"
#include <sstream>

ASRC_NAMESPACE_BEGIN(asrc)
ASRC_NAMESPACE_BEGIN(core)
ASRC_INLINE_NAMESPACE_BEGIN(util)

template<class T>
struct PYBIND11_EXPORT weak_ptr_property_tag {
    typedef boost::vertex_property_tag kind;
};

template<
	class _Instance,
	class _Identifier,
	traits::converter<const std::shared_ptr<const _Instance>&,_Identifier> _IdentifierGetter,
	traits::stringifier<const std::shared_ptr<const _Instance>&> _Stringifier
>
struct PYBIND11_EXPORT DependencyGraph {
	using Instance = _Instance;
	using Identifier = _Identifier;
	using IdentifierGetter=_IdentifierGetter;
    using Stringifier = _Stringifier;
	using Property = boost::property<weak_ptr_property_tag<Instance>,std::weak_ptr<Instance>>;
    using Graph = boost::adjacency_list<boost::listS,boost::listS,boost::bidirectionalS,Property>;
    using Vertex = typename boost::graph_traits<Graph>::vertex_descriptor;
    using Edge = typename boost::graph_traits<Graph>::edge_descriptor;

	IdentifierGetter identify;
    Stringifier stringify;
	Graph graph;
	std::deque<Vertex> topo_order;
	typename boost::property_map<Graph,weak_ptr_property_tag<Instance>>::type property;
	std::map<Identifier,Vertex> reverse_map;
	std::map<Vertex,std::size_t> vertex_to_index;
	bool auto_reorder;
	bool sorted;

	DependencyGraph(){
		property=get(weak_ptr_property_tag<Instance>(),graph);
		auto_reorder=true;
		sorted=true;
	}

	void clear(){
		topo_order.clear();
		reverse_map.clear();
		vertex_to_index.clear();
		graph.clear();
		sorted=true;
	}

	Vertex add_instance(const std::shared_ptr<Instance>& instance){
		assert(instance);
		auto eid=identify(instance);
		assert(reverse_map.find(eid)==reverse_map.end());

		auto ret=add_vertex({instance},graph);
		reverse_map[eid]=ret;
		vertex_to_index[ret]=topo_order.size();
		topo_order.push_back(ret);
		return ret;
	}

	void remove_instance(const std::shared_ptr<Instance>& instance){
		assert(instance);
		auto eid=identify(instance);
		auto found=reverse_map.find(eid);
		assert(found!=reverse_map.end());
		remove_vertex(found->second,graph);
		auto affected=topo_order.erase(std::find(topo_order.begin(),topo_order.end(),found->second));
		reverse_map.erase(found);
		for(auto&& v : std::ranges::subrange(affected,topo_order.end())){
			vertex_to_index[v]--;
		}
	}

	bool contains(const std::shared_ptr<Instance>& instance) const{
		if(!instance){
			return false;
		}
		auto eid=identify(instance);
		auto found=reverse_map.find(eid);
		return found!=reverse_map.end();
	}

	bool has_dependency(const Vertex& src, const Vertex& dst) const{
		return edge(src,dst,graph).second;
	}

	bool has_dependency(const std::shared_ptr<Instance>& src, const std::shared_ptr<Instance>& dst) const{
		return has_dependency(get_vertex(src),get_vertex(dst));
	}

	bool add_dependency(const Vertex& src, const Vertex& dst){
		// 更新されたかどうかを返す
		if(has_dependency(src,dst)){
			return false;
		}else{
			add_edge(src,dst,graph);
			sorted=false;
			if(auto_reorder){
				online_reorder(src,dst);
				//reorder();
			}
			return true;
		}
	}

	bool add_dependency(const std::shared_ptr<Instance>& src, const std::shared_ptr<Instance>& dst){
		return add_dependency(get_vertex(src),get_vertex(dst));
	}

	void remove_dependency(const Vertex& src, const Vertex& dst){
		if(has_dependency(src,dst)){
			remove_edge(src,dst,graph);
		}
	}

	void remove_dependency(const std::shared_ptr<Instance>& src, const std::shared_ptr<Instance>& dst){
		remove_dependency(get_vertex(src),get_vertex(dst));
	}

	bool is_sorted() const {
		return sorted;
	}

	void reorder(){
    	topo_order.clear();

		std::map<Vertex,boost::default_color_type> color_map_storage;
		boost::associative_property_map<decltype(color_map_storage)> cmap(color_map_storage);
    	boost::topological_sort(graph,std::front_inserter(topo_order),color_map(cmap));
		sorted=true;
		std::size_t i=0;
		for(auto&& v : topo_order){
			vertex_to_index[v]=i;
			++i;
		}
	}

	std::shared_ptr<Instance> get_instance(const Vertex& v) const{
		auto weak=boost::get(property,v);
		assert(!weak.expired());
		return weak.lock();
	}

	std::weak_ptr<Instance> get_weak_instance(const Vertex& v) const{
		auto weak=boost::get(property,v);
		assert(!weak.expired());
		return weak;
	}

	Vertex get_vertex(const std::shared_ptr<Instance>& instance) const{
		assert(instance);
		auto eid=identify(instance);
		auto found=reverse_map.find(eid);
		assert(found!=reverse_map.end());
		return found->second;
	}

	void online_reorder(const Vertex& src,const Vertex& tgt){
		// ソート済の状態から、srcからtgtへのEdgeを1本追加して再ソートを行う。
		// 
		// [Kavitha 2007] Kavitha, T. et al. "Faster Algorithms for Online Topological Ordering." arXiv:0711.0251
		if(vertex_to_index[src]<vertex_to_index[tgt]){
			sorted=true;
			return;
		}

		using i_v_pair=std::pair<std::size_t,Vertex>;
		auto i_less=[](const i_v_pair& lhs, const i_v_pair& rhs){
			return lhs.first<rhs.first;
		};
		auto i_greater=[](const i_v_pair& lhs, const i_v_pair& rhs){
			return lhs.first>rhs.first;
		};
		std::set<std::size_t> i_set;
		std::set<i_v_pair,decltype(i_less)> anc(i_less),des(i_less);

		boost::heap::fibonacci_heap<
			i_v_pair,
			boost::heap::compare<decltype(i_less)>
		> Fu(i_less);//最大ヒープ
		boost::heap::fibonacci_heap<
			i_v_pair,
			boost::heap::compare<decltype(i_greater)>
		> Fv(i_greater);//最小ヒープ
		std::unordered_set<Vertex> Fu_visited_set;
		std::unordered_set<Vertex> Fv_visited_set;

		std::size_t ma=in_degree(src,graph);
		std::size_t md=out_degree(tgt,graph);

		double logn=log(topo_order.size());

		i_v_pair x{vertex_to_index[src],src}, y{vertex_to_index[tgt],tgt};
		anc.insert(x);
		i_set.insert(x.first);
		des.insert(y);
		i_set.insert(y.first);

		std::set<i_v_pair,decltype(i_less)> toBeShifted1(i_less),toBeShifted2(i_less);

		while(true){
			bool balanced = ma<=md ? md-ma<=anc.size()*logn : ma-md<=des.size()*logn;
			if(balanced){
				// visit(x)
				for(auto&& w : get_predecessors(x.second)){
					if(!Fu_visited_set.contains(w)){
						if(Fv_visited_set.contains(w)){
							std::ostringstream oss;
							oss<<"cycle detected while adding an edge ("<<stringify(get_instance(src))<<" -> "<<stringify(get_instance(tgt))<<"). cycle=[";
							oss<<stringify(get_instance(w))<<" -> ";
							oss<<stringify(get_instance(x.second))<<" -> { ";
							for(auto&& fu_visited : Fu_visited_set){
								oss<<stringify(get_instance(fu_visited))<<" , ";
							}
							oss<<" } -> { ";
							for(auto&& fv_visited : Fv_visited_set){
								oss<<stringify(get_instance(fv_visited))<<" , ";
							}
							oss<<" }";
							oss<<"]";
							throw std::runtime_error(oss.str());
						}
						Fu.push({vertex_to_index[w],w});
						Fu_visited_set.insert(w);
					}
				}
				// visit(y)
				for(auto&& w : get_successors(y.second)){
					if(!Fv_visited_set.contains(w)){
						if(Fu_visited_set.contains(w)){
							std::ostringstream oss;
							oss<<"cycle detected while adding an edge ("<<stringify(get_instance(src))<<" -> "<<stringify(get_instance(tgt))<<"). cycle=[";
							oss<<stringify(get_instance(w))<<" -> ";
							oss<<stringify(get_instance(x.second))<<" -> { ";
							for(auto&& fu_visited : Fu_visited_set){
								oss<<stringify(get_instance(fu_visited))<<" , ";
							}
							oss<<" } -> { ";
							for(auto&& fv_visited : Fv_visited_set){
								oss<<stringify(get_instance(fv_visited))<<" , ";
							}
							oss<<" }";
							oss<<"]";
							throw std::runtime_error(oss.str());
						}
						Fv.push({vertex_to_index[w],w});
						Fv_visited_set.insert(w);
					}
				}
				// extract-max(Fu)
				if(Fu.size()>0){
					x=Fu.top();
					anc.insert(x);
					i_set.insert(x.first);
					Fu.pop();
				}else{
					// done
					// yの直前にancとdesを挿入
					for(std::size_t i=*i_set.begin()+1;i<y.first;++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted1.insert(z);
							i_set.insert(i);
						}
					}
					for(std::size_t i=y.first+1;i<topo_order.size();++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted2.insert(z);
							i_set.insert(i);
						}
					}
					break;
				}
				// extract-min(Fv)
				if(Fv.size()>0){
					y=Fv.top();
					des.insert(y);
					i_set.insert(y.first);
					Fv.pop();
				}else{
					// done
					// xの直後にancとdesを挿入
					for(std::size_t i=*i_set.begin()+1;i<x.first;++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted1.insert(z);
							i_set.insert(i);
						}
					}
					for(std::size_t i=x.first+1;i<topo_order.size();++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted2.insert(z);
							i_set.insert(i);
						}
					}
					break;
				}
				if(x.first<y.first){
					// done
					// xの直後にancとdesを挿入
					for(std::size_t i=*i_set.begin()+1;i<x.first;++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted1.insert(z);
							i_set.insert(i);
						}
					}
					for(std::size_t i=x.first+1;i<topo_order.size();++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted2.insert(z);
							i_set.insert(i);
						}
					}
					break;
				}else{
					ma+=in_degree(x.second,graph);
					md+=out_degree(y.second,graph);
				}
			}else if(ma<md){
				// visit(x)
				for(auto&& w : get_predecessors(x.second)){
					if(!Fu_visited_set.contains(w)){
						if(Fv_visited_set.contains(w)){
							std::ostringstream oss;
							oss<<"cycle detected while adding an edge ("<<stringify(get_instance(src))<<" -> "<<stringify(get_instance(tgt))<<"). cycle=[";
							oss<<stringify(get_instance(w))<<" -> ";
							oss<<stringify(get_instance(x.second))<<" -> { ";
							for(auto&& fu_visited : Fu_visited_set){
								oss<<stringify(get_instance(fu_visited))<<" , ";
							}
							oss<<" } -> { ";
							for(auto&& fv_visited : Fv_visited_set){
								oss<<stringify(get_instance(fv_visited))<<" , ";
							}
							oss<<" }";
							oss<<"]";
							throw std::runtime_error(oss.str());
						}
						Fu.push({vertex_to_index[w],w});
						Fu_visited_set.insert(w);
					}
				}
				// extract-max(Fu)
				if(Fu.size()>0){
					x=Fu.top();
					anc.insert(x);
					i_set.insert(x.first);
					Fu.pop();
				}else{
					// done
					// yの直前にancとdesを挿入
					for(std::size_t i=*i_set.begin()+1;i<y.first;++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted1.insert(z);
							i_set.insert(i);
						}
					}
					for(std::size_t i=y.first+1;i<topo_order.size();++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted2.insert(z);
							i_set.insert(i);
						}
					}
					break;
				}
				if(x.first<y.first){
					// done
					// xの直後にancとdesを挿入
					for(std::size_t i=*i_set.begin()+1;i<x.first;++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted1.insert(z);
							i_set.insert(i);
						}
					}
					for(std::size_t i=x.first+1;i<topo_order.size();++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted2.insert(z);
							i_set.insert(i);
						}
					}
					break;
				}else{
					ma+=in_degree(x.second,graph);
				}
			}else{
				// visit(y)
				for(auto&& w : get_successors(y.second)){
					if(!Fv_visited_set.contains(w)){
						if(Fu_visited_set.contains(w)){
							std::ostringstream oss;
							oss<<"cycle detected while adding an edge ("<<stringify(get_instance(src))<<" -> "<<stringify(get_instance(tgt))<<"). cycle=[";
							oss<<stringify(get_instance(w))<<" -> ";
							oss<<stringify(get_instance(x.second))<<" -> { ";
							for(auto&& fu_visited : Fu_visited_set){
								oss<<stringify(get_instance(fu_visited))<<" , ";
							}
							oss<<" } -> { ";
							for(auto&& fv_visited : Fv_visited_set){
								oss<<stringify(get_instance(fv_visited))<<" , ";
							}
							oss<<" }";
							oss<<"]";
							throw std::runtime_error(oss.str());
						}
						Fv.push({vertex_to_index[w],w});
						Fv_visited_set.insert(w);
					}
				}
				// extract-min(Fv)
				if(Fv.size()>0){
					y=Fv.top();
					des.insert(y);
					i_set.insert(y.first);
					Fv.pop();
				}else{
					// done
					// xの直後にancとdesを挿入
					for(std::size_t i=*i_set.begin()+1;i<x.first;++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted1.insert(z);
							i_set.insert(i);
						}
					}
					for(std::size_t i=x.first+1;i<topo_order.size();++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted2.insert(z);
							i_set.insert(i);
						}
					}
					break;
				}
				if(x.first<y.first){
					// done
					// yの直前にancとdesを挿入
					for(std::size_t i=*i_set.begin()+1;i<y.first;++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted1.insert(z);
							i_set.insert(i);
						}
					}
					for(std::size_t i=y.first+1;i<topo_order.size();++i){
						if(!i_set.contains(i)){
							i_v_pair z{i,topo_order[i]};
							toBeShifted2.insert(z);
							i_set.insert(i);
						}
					}
					break;
				}else{
					md+=out_degree(y.second,graph);
				}
			}
		}

		auto it=i_set.begin();
		for(auto&& w : toBeShifted1){
			vertex_to_index[w.second]=*it;
			topo_order[*it]=w.second;
			++it;
		}
		for(auto&& w : anc){
			vertex_to_index[w.second]=*it;
			topo_order[*it]=w.second;
			++it;
		}
		for(auto&& w : des){
			vertex_to_index[w.second]=*it;
			topo_order[*it]=w.second;
			++it;
		}
		for(auto&& w : toBeShifted2){
			vertex_to_index[w.second]=*it;
			topo_order[*it]=w.second;
			++it;
		}
		sorted=true;
	}

	template<std::derived_from<Instance> Derived>
	Vertex get_vertex(const std::weak_ptr<Derived>& instance) const{
		assert(!instance.expired());
		return get_vertex(instance.lock());
	}

	Vertex get_vertex(const std::size_t& index){
		if(!sorted){
			reorder();
		}
		assert(index>=0 && index<topo_order.size());
		return topo_order.at(index);
	}

	std::size_t get_index(const Vertex& v){
		if(!sorted){
			reorder();
		}
		auto found=vertex_to_index.find(v);
		assert(found!=vertex_to_index.end());
		return found->second;
	}

	template<std::derived_from<Instance> Derived>
	std::size_t get_index(const std::shared_ptr<Derived>& instance){
		assert(instance);
		auto eid=identify(instance);
		auto found=reverse_map.find(eid);
		assert(found!=reverse_map.end());
		return get_index(found->second);
	}

	template<std::derived_from<Instance> Derived>
	std::size_t get_index(const std::weak_ptr<Derived>& instance){
		assert(!instance.expired());
		return get_index(instance.lock());
	}

	auto get_range(){
		if(!sorted){
			reorder();
		}
		return (
			topo_order
			| std::views::transform([this](const Vertex& v){return get_instance(v);})
		);
	}

	auto get_weak_range(){
		if(!sorted){
			reorder();
		}
		return (
			topo_order
			| std::views::transform([this](const Vertex& v){return get_weak_instance(v);})
		);
	}

	auto get_predecessors(const Vertex& v) const{
		auto [in_begin, in_end] = in_edges(v,graph);
		return (
			std::ranges::subrange(in_begin,in_end)
			| std::views::transform([this](const Edge& e){return source(e,graph);})
		);
	}

	template<std::derived_from<Instance> Derived>
	auto get_predecessors(const std::shared_ptr<Derived>& instance) const{
		assert(instance);
		auto eid=identify(instance);
		auto found=reverse_map.find(eid);
		assert(found!=reverse_map.end());
		return (
			get_predecessors(found->second)
			| std::views::transform([this](const Vertex& v){return get_instance(v);})
		);
	}

	template<std::derived_from<Instance> Derived>
	auto get_predecessors(const std::weak_ptr<Derived>& instance) const{
		assert(!instance.expired());
		return get_predecessors(instance.lock());
	}

	auto get_predecessors_by_index(const std::size_t& i){
		if(!sorted){
			reorder();
		}
		return (
			get_predecessors(topo_order.at(i))
			| std::views::transform([this](const Vertex& v){return vertex_to_index[v];})
		);
	}

	auto get_successors(const Vertex& v) const{
		auto [out_begin, out_end] = out_edges(v,graph);
		return (
			std::ranges::subrange(out_begin,out_end)
			| std::views::transform([this](const Edge& e){return target(e,graph);})
		);
	}

	template<std::derived_from<Instance> Derived>
	auto get_successors(const std::shared_ptr<Derived>& instance) const{
		assert(instance);
		auto eid=identify(instance);
		auto found=reverse_map.find(eid);
		assert(found!=reverse_map.end());
		return (
			get_successors(found->second)
			| std::views::transform([this](const Vertex& v){return get_instance(v);})
		);
	}

	template<std::derived_from<Instance> Derived>
	auto get_successors(const std::weak_ptr<Derived>& instance) const{
		assert(!instance.expired());
		return get_successors(instance.lock());
	}

	auto get_successors_by_index(const std::size_t& i){
		if(!sorted){
			reorder();
		}
		return (
			get_successors(topo_order.at(i))
			| std::views::transform([this](const Vertex& v){return vertex_to_index[v];})
		);
	}

    template<class Archive>
    void save(Archive & archive) const{
		std::vector<std::shared_ptr<Instance>> instances;
		instances.reserve(topo_order.size());
		for(auto&& v : topo_order){
			instances.push_back(get_instance(v));
		}
		std::vector<std::pair<std::size_t,std::size_t>> dependencies;
		auto es=edges(graph);
		for(auto&& e : std::ranges::subrange(es.first,es.second)){
			dependencies.push_back({
				vertex_to_index.at(source(e,graph)),
				vertex_to_index.at(target(e,graph))
			});
		}

        archive(
            CEREAL_NVP(instances),
            CEREAL_NVP(dependencies),
            CEREAL_NVP(sorted),
            CEREAL_NVP(auto_reorder)
		);
	}

    template<class Archive>
    void load(Archive & archive){
		auto_reorder=false;

		std::vector<std::shared_ptr<Instance>> instances;
        archive(
            CEREAL_NVP(instances)
		);

		clear();
		for(auto&& instance : instances){
			add_instance(instance);
		}
	
		std::vector<std::pair<std::size_t,std::size_t>> dependencies;
        archive(
            CEREAL_NVP(dependencies)
		);

		for(auto&& [pre,suc] : dependencies){
			add_dependency(topo_order[pre],topo_order[suc]);
		}

        archive(
            CEREAL_NVP(sorted),
            CEREAL_NVP(auto_reorder)
		);
	}

	void debug_print(){
		std::cout<<"["<<std::endl;
		for(auto&& v : get_range()){
			std::cout<<"  "<<stringify(v)<<std::endl;
		}
		std::cout<<"]"<<std::endl;
	}
};

ASRC_NAMESPACE_END(util)

ASRC_NAMESPACE_BEGIN(traits)
template<class T>
struct is_dependency_graph : std::false_type {};

template<
	class Instance,
	class Identifier,
	converter<const std::shared_ptr<const Instance>&,Identifier> IdentifierGetter,
	stringifier<const std::shared_ptr<const Instance>&> Stringifier
>
struct is_dependency_graph<util::DependencyGraph<Instance,Identifier,IdentifierGetter,Stringifier>> : std::true_type {};

template<class T>
concept dependency_graph = is_dependency_graph<T>::value;

ASRC_NAMESPACE_END(traits)

ASRC_INLINE_NAMESPACE_BEGIN(util)

template<traits::dependency_graph GraphType, typename ... Extras>
pybind11::class_<GraphType> expose_dependency_graph_name(pybind11::module &m,const std::string& className, Extras&& ... extras){

    using RangeType = decltype(std::declval<GraphType>().get_range());
    using WeakRangeType = decltype(std::declval<GraphType>().get_weak_range());

	expose_range<RangeType,Extras...>(m,std::forward<Extras>(extras)...);	
	expose_range<WeakRangeType,Extras...>(m,std::forward<Extras>(extras)...);

	using Instance = typename GraphType::Instance;

	try{
		return pybind11::class_<GraphType>(m,className.c_str(),std::forward<Extras>(extras)...)
		.def(py_init<>())
		.def("add_instance",&GraphType::add_instance)
		.def("remove_instance",&GraphType::remove_instance)
		.def("contains",&GraphType::contains)
		.def("has_dependency",pybind11::overload_cast<const std::shared_ptr<Instance>&,const std::shared_ptr<Instance>&>(&GraphType::has_dependency,pybind11::const_))
		.def("add_dependency",pybind11::overload_cast<const std::shared_ptr<Instance>&,const std::shared_ptr<Instance>&>(&GraphType::add_dependency))
		.def("remove_dependency",pybind11::overload_cast<const std::shared_ptr<Instance>&,const std::shared_ptr<Instance>&>(&GraphType::remove_dependency))
		.def("reorder",&GraphType::reorder)
		.def("get_instance",&GraphType::get_instance)
		.def("get_weak_instance",&GraphType::get_weak_instance)
		.def("get_range",&GraphType::get_range,pybind11::return_value_policy::reference_internal)
		.def("get_weak_range",&GraphType::get_weak_range,pybind11::return_value_policy::reference_internal)
		;
    }catch(std::exception& ex){
        if(pybind11::detail::get_type_info(typeid(GraphType))){
            // already bound
            auto ret=pybind11::reinterpret_borrow<pybind11::class_<GraphType>>(pybind11::detail::get_type_handle(typeid(GraphType),true));
            assert(ret);
            return std::move(ret);
        }else{
            throw ex;
        }
	}
}

template<traits::dependency_graph GraphType, typename ... Extras>
pybind11::class_<GraphType> expose_dependency_graph(pybind11::module &m, Extras&& ... extras){
	return expose_dependency_graph_name<GraphType,Extras...>(m,cereal::util::demangle(typeid(GraphType).name()),std::forward<Extras>(extras)...);
}

ASRC_NAMESPACE_END(util)
ASRC_NAMESPACE_END(core)
ASRC_NAMESPACE_END(asrc)
