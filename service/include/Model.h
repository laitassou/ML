/*
* class Model.h
* user: laitassou
* description
*/

#pragma once

#include <cstring>
#include <algorithm>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <tuple>
#include <memory>
#include <tensorflow/c/c_api.h>
#include "Tensor.h"

class Tensor;


class Model {
public:
    struct GraphCreate {
        TF_Graph * operator()() { return TF_NewGraph();}
    };
    struct GraphDeleter {
        void operator()(TF_Graph* b) { TF_DeleteGraph(b);}
    };
    
    struct SessionDeleter {
        void operator()(TF_Session* sess, TF_Status * status ) { TF_DeleteSession(sess,status);}
        
    };
    
    struct StatusDeleter {
        void operator()(TF_Status* status) { TF_DeleteStatus(status);}        
    };


    using unique_graph_ptr = std::unique_ptr<TF_Graph, GraphDeleter>;
    using unique_session_ptr = std::unique_ptr<TF_Session, SessionDeleter>;
    using unique_status_ptr = std::unique_ptr<TF_Status, StatusDeleter >;
    explicit Model(const std::string&);

    // Rule of five, moving is easy as the pointers can be copied, copying not as i have no idea how to copy
    // the contents of the pointer (i guess dereferencing won't do a deep copy)
    Model(const Model &model) = delete;
    Model(Model &&model) = default;
    Model& operator=(const Model &model) = delete;
    Model& operator=(Model &&model) = default;

    ~Model();

    void init();
    void restore(const std::string& ckpt);
    void save(const std::string& ckpt);
    std::vector<std::string> get_operations() const;

    // Original Run
    void run(const std::vector<Tensor*>& inputs, const std::vector<Tensor*>& outputs);

    // Run with references
    void run(Tensor& input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor& output);
    void run(Tensor& input, Tensor& output);

    // Run with pointers
    void run(Tensor* input, const std::vector<Tensor*>& outputs);
    void run(const std::vector<Tensor*>& inputs, Tensor* output);
    void run(Tensor* input, Tensor* output);

private:
    unique_graph_ptr _graph;
    //unique_session_ptr _session;
    TF_Session * _session;
    unique_status_ptr _status;

    // Read a file from a string
    static TF_Buffer* read(const std::string&);

    bool status_check(bool throw_exc) const;
    void error_check(bool condition, const std::string &error) const;

public:
    friend class Tensor;
};


