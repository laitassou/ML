#include "../include/ModelInfos.hpp"


template <class T> ModelInfos<T>::ModelInfos(const std::string & name):input_depth(0)
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


template <class T> const T &  ModelInfos<T>::get_mean(const std::string &label) const
{
    auto it = labels.find(label);
    if(it == labels.end())
    {
        std::cout <<"label: "<< label << "not found" <<std::endl;
        return T(0);
    }
    else{
        return it->second.mean;
    }        
}

template <class T> const T &  ModelInfos<T>::get_std(const std::string &label) const
{
    auto it = labels.find(label);
    if(it == labels.end())
    {
        std::cout <<"label: "<< label << "not found" <<std::endl;
        return T(0);
    }
    else{
        return it->second.std;
    }        
}

template <class T> void ModelInfos<T>::show(void) const
{
    using boost::property_tree::ptree;
    ptree::const_iterator end = pt_json.end();
    for (ptree::const_iterator it = pt_json.begin(); it != end; ++it) {
        std::cout << it->first << " :" <<std::endl;
        auto row = it->second;
        for(auto & rr :row)
            std::cout << rr.first << ":" << rr.second.data()<< std::endl;
    }

    // show map
    for(  auto  itr =labels.begin(); itr != labels.end(); ++itr)
    {
        std::cout << "label: " <<itr->first << "\n";
        auto row = itr->second;

        std::cout << "stats:  "<< itr->second.std << ","<<  itr->second.mean << std::endl;
    }
}


template <class T> void ModelInfos<T>::parse(void) 
{
    using boost::property_tree::ptree;
    ptree::const_iterator end = pt_json.end();
    for (ptree::const_iterator it = pt_json.begin(); it != end; ++it) 
    {
        std::cout << it->first << " :" <<std::endl;    

        auto row = it->second;
        if ((it->first).compare("mean")==0)
        {

            for(auto & rr :row)
            {
                struct Stats stat {0,0};
                std::cout << rr.first << ":" << rr.second.data()<< std::endl;

                if (labels.find(rr.first) == labels.end())
                {
                    stat.mean = std::stod(rr.second.data());
                    labels.insert(std::pair<std::string, struct Stats>(rr.first,stat));
                }
                else
                {
                    stat = labels[rr.first] ;
                    stat.std = std::stod(rr.second.data());
                    labels[rr.first]  = stat;
                }
                stat.mean = std::stod( rr.second.data());
                labels.insert({rr.first,stat});
            }
        }
        else if ((it->first).compare("std")==0)
        {
            for(auto & rr :row){
                struct Stats stat {0,0};
                std::cout << rr.first << ":" << rr.second.data()<< std::endl;
                if (labels.find(rr.first) == labels.end())
                {
                    stat.std= std::stod(rr.second.data());;
                    labels.insert(std::pair<std::string, struct Stats>(rr.first,stat));
                }
                else
                {
                    stat = labels[rr.first] ;
                    stat.std = std::stod(rr.second.data());
                    labels[rr.first]  = stat;
                }
            }
        }          
            
    }
}


template <class T> void ModelInfos<T>::compute_columns(void) 
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


template class  ModelInfos<float>;
template class  ModelInfos<double>;
template class  ModelInfos<int>;
template class  ModelInfos<std::string>;