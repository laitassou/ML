#pragma once 

#include <iostream>


template <class T > class ModelManager {
    public:
    
    virtual ~ModelManager() = default;
    virtual std::vector<uint32_t> AvailableModels() const = 0;

    private:

    struct ModelId{
        uint32_t _id;
        std::string _name 
    }
    std::map<ModelId, class T > list_models;

};