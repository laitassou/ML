/*
* class ModelInfos.h
* user: laitassou
* description
*/

#pragma once


#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <boost/foreach.hpp>
#include <cassert>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <map>


template <class T> class ModelInfos
{
    public:
    struct Stats
    {
        T mean;
        T std;
    };
    using json_ptree= boost::property_tree::ptree;

    ModelInfos(const std::string & name);
    const T & get_mean(const std::string &label) const;
    const T & get_std(const std::string &label) const;
    void show(void) const;
    void compute_columns(void);
    void parse(void);
    inline size_t get_depth()const
    {
        return input_depth;
    };

    private:
    size_t input_depth;
    std::map<std::string,struct Stats> labels;
    json_ptree  pt_json;
};