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

    ModelInfos(const std::string & name):input_depth(0)
    {

        try
        {
            //std::stringstream ss;
            // send your JSON above to the parser below, but populate ss first
            //boost::property_tree::ptree pt;

            std::ifstream jsonFile(name);
            boost::property_tree::read_json(jsonFile, pt_json);
 
        }
        catch (std::exception const& e)
        {
            std::cout<< e.what()<< std::endl;
        }
    }
    const std::string & get(const std::string &label) const
    {
        auto it = labels.find(label);
        if(it == labels.end())
        {
            return std::string();
        }
        else{
            return it->second;
        }        
    }

    size_t get_depth()const
    {
        return input_depth;
    }

    void show(void) const
    {
        using boost::property_tree::ptree;
        ptree::const_iterator end = pt_json.end();
        for (ptree::const_iterator it = pt_json.begin(); it != end; ++it) {
            std::cout << it->first << " :" <<std::endl;
            auto row = it->second;
            for(auto & rr :row)
                std::cout << rr.first << ":" << rr.second.data()<< std::endl;
        }
    }
     void parse(void) 
    {
        using boost::property_tree::ptree;
        ptree::const_iterator end = pt_json.end();
        for (ptree::const_iterator it = pt_json.begin(); it != end; ++it) {
            auto row = it->second;
            for(auto & rr :row)
            {
                input_depth++;
                std::cout << rr.first << ", input_depth :" << input_depth <<std::endl;
            }
            break;
            
        }
    }  

    private:
    size_t input_depth;
    std::map<std::string,struct Stats> labels;
    boost::property_tree::ptree  pt_json;
};