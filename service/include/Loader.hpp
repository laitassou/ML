#praga once


namespace ML {
enum class Satus
{
    LOAD_STARTED,
    LOAD_ONGOING,
    LOAD_ENDED
}

class Loader{
 public:
     
     void Load() 
     {
         doLoad();
     };


     void Unload()
     { 
         doUnload();
     }


     void EstimateRessource()
     {
       doEstimate();
    }


 private:    

   virtual void doLoad() =0;
   virtual void doUnload() =0;
   virtual void doEstimate() =0;

   std::string _resource_name;


};

using LoaderSource = Source<std::unique_ptr<Loader>>;

}