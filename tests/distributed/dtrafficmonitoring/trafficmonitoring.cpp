
#include <regex>
#include <string>
#include <vector>
#include <iostream>
#include <ff/dff.hpp>
#include <getopt.h>
#include "geo_model.hpp"


using namespace std;
using namespace ff;

/// application run time (source generates the stream for app_run_time seconds, then sends out EOS)
unsigned long app_run_time = 60 * 1000000000L; // 60 seconds
const size_t qlen = 2048;


/// components and topology name
const string topology_name = "TrafficMonitoring";
const string source_name = "source";
const string map_match_name = "map_matcher";
const string speed_calc_name = "speed_calculator";
const string sink_name = "sink";

typedef enum { BEIJING, DUBLIN } city;

/// information contained in each record in the Beijing dataset
typedef enum { TAXI_ID_FIELD, NID_FIELD, DATE_FIELD, TAXI_LATITUDE_FIELD, TAXI_LONGITUDE_FIELD,
               TAXI_SPEED_FIELD, TAXI_DIRECTION_FIELD } beijing_record_field;

/// information contained in each record in the Dublin dataset
typedef enum { TIMESTAMP_FIELD, LINE_ID_FIELD, BUS_DIRECTION_FIELD, JOURNEY_PATTERN_ID_FIELD, TIME_FRAME_FIELD,
               VEHICLE_JOURNEY_ID_FIELD, OPERATOR_FIELD, CONGESTION_FIELD, BUS_LONGITUDE_FIELD, BUS_LATITUDE_FIELD,
               DELAY_FIELD, BLOCK_ID_FIELD, BUS_ID_FIELD, STOP_ID_FIELD, AT_STOP_ID_FIELD } dublin_record_field;

/// Beijing bounding box
const double beijing_lat_min = 39.689602;
const double beijing_lat_max = 40.122410;
const double beijing_lon_min = 116.105789;
const double beijing_lon_max = 116.670021;

/// Dublin bounding box
const double dublin_lat_min = 53.28006;
const double dublin_lat_max = 53.406071;
const double dublin_lon_min = -6.381911;
const double dublin_lon_max = -6.141994;

/// application parameters
city _monitored_city = BEIJING;     // user can choose between two options: BEIJING and DUBLIN

const string _beijing_input_file = "dataset/taxi-traces.csv";           // path of the Beijing dataset to be used
const string _dublin_input_file = "../data/bus-traces_20130101.csv";    // path of the Dublin dataset to be used

const string _beijing_shapefile = "dataset/beijing/roads.shp";          // path of the Beijing shape file
const string _dublin_shapefile = "../data/dublin/roads.shp";            // path of the Dublin shape file

size_t _road_win_size = 1000;

struct tuple_t {
    double latitude;            // vehicle latitude
    double longitude;           // vehicle longitude
    double speed;               // vehicle speed
    int direction;              // vehicle direction
    size_t key;                 // vehicle_id that identifies the vehicle (taxi or bus)
    uint64_t ts;

    // default constructor
    tuple_t() : latitude(0.0), longitude(0.0), speed(0.0), direction(0), key(0) {}

    // constructor
    tuple_t(double _latitude, double _longitude, double _speed, int _direction, size_t _key) :
        latitude(_latitude), longitude(_longitude), speed(_speed), direction(_direction), key(_key) {}

    template<class Archive>
	void serialize(Archive & archive) {
		archive(latitude,longitude,speed, direction, key, ts);
	}
};

struct result_t {
    double speed;               // vehicle speed
    size_t key;                 // road ID corresponding to latitude and longitude coordinates of the vehicle
    uint64_t ts;

    // default constructor
    result_t(): speed(0.0), key(0) {}

    // constructor
    result_t(double _speed, size_t _key, uint64_t _id, uint64_t _ts): speed(_speed), key(_key) {}

    template<class Archive>
	void serialize(Archive & archive) {
		archive(speed,key,ts);
	}
};

static inline unsigned long current_time_usecs() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000L + (t.tv_nsec / 1000);
}
static inline unsigned long current_time_nsecs() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}

// Type of Beijing input records:
// < vehicle_ID_value, n_ID_value, date_value, latitude_value, longitude_value, speed_value, direction_value >
using beijing_record_t = tuple<int, int, string, double, double, double, int>;

// Type of Dublin input records:
// < timestamp_value, line_ID_value, direction_value, journey_pattern_ID_value, time_frame_value, vehicle_journey_ID_value, operator_value,
// congestion_value, longitude_value, latitude_value, delay_value, block_ID_value, bus_ID_value, stop_ID_value, at_stop_value >
using dublin_record_t = tuple<long, int, int, string, string, int, string, int, double, double, int, int, int, int, int>;

// global variables
vector<beijing_record_t> beijing_parsed_file;           // contains data extracted from the Beijing input file
vector<dublin_record_t> dublin_parsed_file;             // contains data extracted from the Dublin input file
vector<tuple_t> dataset;                                // contains all the tuples in memory
unordered_map<size_t, uint64_t> key_occ;                // contains the number of occurrences of each key vehicle_id
Road_Grid_List road_grid_list;                          // contains data extracted from the city shapefile
atomic<long> sent_tuples;                               // total number of tuples sent by all the sources

/** 
 *  @brief Parse the input file.
 *  
 *  The file is parsed and saved in memory.
 *  @param file_path the path of the input dataset file
 */ 
void parse_dataset(const string& file_path) {
    ifstream file(file_path);
    if (file.is_open()) {
        size_t all_records = 0;         // counter of all records (dataset line) read
        size_t incomplete_records = 0;  // counter of the incomplete records
        string line;
        while (getline(file, line)) {
            // process file line
            int token_count = 0;
            vector<string> tokens;
            size_t last = 0;
            size_t next = 0;
            while ((next = line.find(',', last)) != string::npos) {
                tokens.push_back(line.substr(last, next - last));
                last = next + 1;
                token_count++;
            }
            tokens.push_back(line.substr(last));
            token_count++;
            // A record is valid if it contains at least 7 values (one for each field of interest)
            // in the case in which the application analyzes data coming from Beijing taxi-traces.
            if (_monitored_city == BEIJING) {
                if (token_count >= 7) {
                    // save parsed file
                    beijing_record_t r(atoi(tokens.at(TAXI_ID_FIELD).c_str()),
                                       atoi(tokens.at(NID_FIELD).c_str()),
                                       tokens.at(DATE_FIELD),
                                       atof(tokens.at(TAXI_LATITUDE_FIELD).c_str()),
                                       atof(tokens.at(TAXI_LONGITUDE_FIELD).c_str()),
                                       atof(tokens.at(TAXI_SPEED_FIELD).c_str()),
                                       atoi(tokens.at(TAXI_DIRECTION_FIELD).c_str()));
                    beijing_parsed_file.push_back(r);

                    // insert the key device_id in the map (if it is not present)
                    if (key_occ.find(get<TAXI_ID_FIELD>(r)) == key_occ.end()) {
                        key_occ.insert(make_pair(get<TAXI_ID_FIELD>(r), 0));
                    }
                }
                else
                    incomplete_records++;
            }
            else if (_monitored_city == DUBLIN) {
                // A record is valid if it contains at least 15 values (one for each field of interest)
                // in the case in which the application analyzes data coming from Dublin bus-traces.
                if (token_count >= 15) {
                    // save parsed file
                    dublin_record_t r(atol(tokens.at(TIMESTAMP_FIELD).c_str()),
                                      atoi(tokens.at(LINE_ID_FIELD).c_str()),
                                      atoi(tokens.at(BUS_DIRECTION_FIELD).c_str()),
                                      tokens.at(JOURNEY_PATTERN_ID_FIELD),
                                      tokens.at(TIME_FRAME_FIELD),
                                      atoi(tokens.at(VEHICLE_JOURNEY_ID_FIELD).c_str()),
                                      tokens.at(OPERATOR_FIELD),
                                      atoi(tokens.at(CONGESTION_FIELD).c_str()),
                                      atof(tokens.at(BUS_LONGITUDE_FIELD).c_str()),
                                      atof(tokens.at(BUS_LATITUDE_FIELD).c_str()),
                                      atoi(tokens.at(DELAY_FIELD).c_str()),
                                      atoi(tokens.at(BLOCK_ID_FIELD).c_str()),
                                      atoi(tokens.at(BUS_ID_FIELD).c_str()),
                                      atoi(tokens.at(STOP_ID_FIELD).c_str()),
                                      atoi(tokens.at(AT_STOP_ID_FIELD).c_str()));
                    dublin_parsed_file.push_back(r);

                    // insert the key device_id in the map (if it is not present)
                    if (key_occ.find(get<BUS_ID_FIELD>(r)) == key_occ.end()) {
                        key_occ.insert(make_pair(get<BUS_ID_FIELD>(r), 0));
                    }
                }
                else
                    incomplete_records++;
            }

            all_records++;
        }
        file.close();
        //if (_monitored_city == BEIJING) print_taxi_parsing_info(beijing_parsed_file, all_records, incomplete_records);
        //else if (_monitored_city == DUBLIN) print_bus_parsing_info(dublin_parsed_file, all_records, incomplete_records);
    }
}

/** 
 *  @brief Process parsed data and create all the tuples.
 *  
 *  The created tuples are maintained in memory. The source node will generate the stream by
 *  reading all the tuples from main memory.
 */ 
void create_tuples() {
    if (_monitored_city == BEIJING) {
        for (int next_tuple_idx = 0; next_tuple_idx < beijing_parsed_file.size(); next_tuple_idx++) {
            // create tuple
            beijing_record_t record = beijing_parsed_file.at(next_tuple_idx);
            tuple_t t;
            t.latitude = get<TAXI_LATITUDE_FIELD>(record);
            t.longitude = get<TAXI_LONGITUDE_FIELD>(record);
            t.speed = get<TAXI_SPEED_FIELD>(record);
            t.direction = get<TAXI_DIRECTION_FIELD>(record);
            t.key = get<TAXI_ID_FIELD>(record);
            //t.id = (key_occ.find(get<TAXI_ID_FIELD>(record)))->second++;
            //t.ts = 0L;
            dataset.insert(dataset.end(), t);
        }
    }
    else if (_monitored_city == DUBLIN) {
        for (int next_tuple_idx = 0; next_tuple_idx < dublin_parsed_file.size(); next_tuple_idx++) {
            // create tuple
            dublin_record_t record = dublin_parsed_file.at(next_tuple_idx);
            tuple_t t;
            t.latitude = get<BUS_LATITUDE_FIELD>(record);
            t.longitude = get<BUS_LONGITUDE_FIELD>(record);
            t.speed = 0.0; // speed values are not present in the used dataset
            t.direction = get<BUS_DIRECTION_FIELD>(record);
            t.key = get<BUS_ID_FIELD>(record);
            //t.id = (key_occ.find(get<BUS_ID_FIELD>(record)))->second++;
            //t.ts = 0L;
            dataset.insert(dataset.end(), t);
        }
    }
}

/** 
 *  @brief Parse the shapefile and create a the Road_Grid_List data structure.
 *  
 *  The data structure containing the processed information about the roads of the city
 *  is passed to the MapMatcher node and use to implement the map matching logic.
 */ 
void read_shapefile() {
    string shapefile_path = (_monitored_city == DUBLIN) ? _dublin_shapefile : _beijing_shapefile;
    if (road_grid_list.read_shapefile(shapefile_path) == -1)
        __throw_invalid_argument("Failed reading shapefile");
}

/**
 * SOURCE NODE 
 **/

struct Source : ff_monode_t<tuple_t> {
    int rate = 0;  
    size_t next_tuple_idx = 0;          // index of the next tuple to be sent
    int generations = 0;                // counts the times the file is generated
    long generated_tuples = 0;          // tuples counter

     // time variables
    unsigned long app_start_time;   // application start time
    unsigned long current_time;

    Source(int rate) : rate(rate) {}
    
    void active_delay(unsigned long waste_time) {
        auto start_time = current_time_nsecs();
        bool end = false;
        while (!end) {
            auto end_time = current_time_nsecs();
            end = (end_time - start_time) >= waste_time;
        }
    }

    int svc_init(){
        app_start_time = current_time_nsecs();
        return 0;
    }

    tuple_t* svc(tuple_t*){
        current_time = current_time_nsecs(); // get the current time
    	// generation loop
    	while (current_time - app_start_time <= app_run_time){
    		if (next_tuple_idx == 0) {
    			generations++;
    		}
    		tuple_t* t = new tuple_t(dataset.at(next_tuple_idx));
    		t->ts = current_time_nsecs();
    		ff_send_out(t); // send the next tuple
    		generated_tuples++;
    		next_tuple_idx = (next_tuple_idx + 1) % dataset.size();   // index of the next tuple to be sent (if any)
	        if (rate != 0) { // active waiting to respect the generation rate
	            long delay_nsec = (long) ((1.0 / rate) * 1e9);
	            active_delay(delay_nsec);
	        }
	        current_time = current_time_nsecs(); // get the new current time
    	}
        return EOS;
    }

    void svc_end(){
        sent_tuples.fetch_add(generated_tuples); // save the number of generated tuples
    }

};

/**
 * MAP MATCHER
 **/


struct MapMatcher : ff_monode_t<tuple_t, result_t> {
    size_t processed = 0;                            // counter of processed tuples
    size_t valid_points = 0;                         // counter of tuples containing GPS coordinates (points) laying inside the city bounding box
    size_t emitted = 0;                              // counter of tuples containing points that correspond to a valid road_id
    Road_Grid_List road_grid_list;               // object containing all the geometric features of the shapefile and used to do map matching
    unordered_map<size_t, uint64_t> key_occ;     // contains the number of occurrences of each key road_id

    // city bounding box
    double max_lon;
    double min_lon;
    double max_lat;
    double min_lat;

    int next_stage_par_deg = 0;

    MapMatcher(Road_Grid_List& _road_grid_list) : road_grid_list(_road_grid_list){
        max_lon = (_monitored_city == DUBLIN) ? dublin_lon_max : beijing_lon_max;
        min_lon = (_monitored_city == DUBLIN) ? dublin_lon_min : beijing_lon_min;
        max_lat = (_monitored_city == DUBLIN) ? dublin_lat_max : beijing_lat_max;
        min_lat = (_monitored_city == DUBLIN) ? dublin_lat_min : beijing_lat_min;
    }

    int svc_init(){
        next_stage_par_deg = this->get_num_outchannels();
        return 0;
    }

    result_t* svc(tuple_t* t){
        if (t->speed >= 0 && t->longitude <= max_lon && t->longitude >= min_lon && t->latitude <= max_lat && t->latitude >= min_lat){
            OGRPoint p(t->longitude, t->latitude);
            int road_id = road_grid_list.fetch_road_ID(p);
            if (road_id != -1) {
                // road_id keys
                if (key_occ.find(road_id) == key_occ.end())
                    key_occ.insert(make_pair(road_id, 0));

                result_t* r = new result_t;
                r->speed = t->speed;
                r->key = road_id;
                r->ts = t->ts;
                ff_send_out_to(r, road_id % next_stage_par_deg);
                emitted++;

            }
            valid_points++;
        }
        processed++;
        delete t;
        return GO_ON;
    }

};

/**
 * SPEED CALCULATOR
 **/

struct SpeedCalculator : ff_minode_t<result_t> {
    
    struct Road_Speed {
        int road_id;
        deque<double> road_speeds;
        size_t win_size;
        double current_sum;
        double incremental_average;
        double squared_sum;
        double incremental_variance;
        size_t count_absolute;

        Road_Speed(int _road_id, double _speed): road_id(_road_id), current_sum(_speed), incremental_average(_speed), squared_sum(_speed * _speed), incremental_variance(0.0), win_size(_road_win_size), count_absolute(0){
            road_speeds.push_back(_speed);
        }

        void update_average_speed(double speed) {
            // control window size
            if (road_speeds.size() > win_size - 1) {
                current_sum -= road_speeds.at(0);
                road_speeds.pop_front();
            }

            // update average speed value
            if (road_speeds.size() == 1) {
                road_speeds.push_back(speed);
                current_sum += speed;
                incremental_average = current_sum / road_speeds.size();
                squared_sum += (speed * speed);
                incremental_variance = squared_sum - (road_speeds.size() * incremental_average * incremental_average);
            } else {
                double cur_avg = (current_sum + speed) / road_speeds.size() + 1;
                double cur_var = (squared_sum + speed * speed) - (road_speeds.size() + 1) * cur_avg * cur_avg;
                double standard_deviation = sqrt(cur_var / road_speeds.size() + 1);

                if (abs(speed - cur_avg) <= 2 * standard_deviation) {
                    road_speeds.push_back(speed);
                    current_sum += speed;
                    squared_sum += (speed * speed);
                    incremental_average = cur_avg;
                    incremental_variance = cur_var;
                }
            }
        }

        ~Road_Speed() {}
    };

    size_t processed = 0;       // tuples counter
    unordered_map<int, Road_Speed> roads;

    result_t* svc(result_t* r){

        if (roads.find(r->key) == roads.end()) {
            Road_Speed rs(r->key, r->speed);
            roads.insert(make_pair(r->key, rs));
        } else {
            roads.at(r->key).update_average_speed(r->speed);
        }

        r->speed = roads.at(r->key).incremental_average;
        processed++;
        return r;
    }
};

/**
 * SINK
 **/

struct Sink : ff_minode_t<result_t> {
    size_t processed = 0;
    result_t* svc(result_t* r){
            processed++;        // tuples counter
            delete r;
            return GO_ON;
    }
};




//opt 

typedef enum { NONE, REQUIRED } opt_arg;    // an option can require one argument or none

const struct option long_opts[] = {
        {"help", NONE, 0, 'h'},
        {"rate", REQUIRED, 0, 'r'},      // pipe start (source) parallelism degree
        {"sampling", REQUIRED, 0, 's'},   // predictor parallelism degree
        {"batch", REQUIRED, 0, 'b'},        // pipe end (sink) parallelism degree
        {"parallelism", REQUIRED, 0, 'p'},        // pipe end (sink) parallelism degree
        {"chaining", NONE, 0, 'c'},
        {0, 0, 0, 0}
};

// Main
int main(int argc, char* argv[]) {

    DFF_Init(argc, argv);

    /// parse arguments from command line
    int option = 0;
    int index = 0;
    
    string file_path;
    size_t source_par_deg = 0;
    size_t matcher_par_deg = 0;
    size_t calculator_par_deg = 0;
    size_t sink_par_deg = 0;
    int rate = 0;
    sent_tuples = 0;
    long sampling = 0;
    bool chaining = false;
    size_t batch_size = 0;
    if (argc == 9 || argc == 10) {
        while ((option = getopt_long(argc, argv, "r:s:p:b:c:", long_opts, &index)) != -1) {
            file_path = _beijing_input_file;
            switch (option) {
                case 'r': {
                    rate = atoi(optarg);
                    break;
                }
                case 's': {
                    sampling = atoi(optarg);
                    break;
                }
                case 'b': {
                    batch_size = atoi(optarg);
                    break;
                }
                case 'p': {
                    vector<size_t> par_degs;
                    string pars(optarg);
                    stringstream ss(pars);
                    for (size_t i; ss >> i;) {
                        par_degs.push_back(i);
                        if (ss.peek() == ',')
                            ss.ignore();
                    }
                    if (par_degs.size() != 4) {
                        printf("Error in parsing the input arguments\n");
                        exit(EXIT_FAILURE);
                    }
                    else {
                        source_par_deg = par_degs[0];
                        matcher_par_deg = par_degs[1];
                        calculator_par_deg = par_degs[2];
                        sink_par_deg = par_degs[3];
                    }
                    break;
                }
                case 'c': {
                    chaining = true;
                    break;
                }
                default: {
                    printf("Error in parsing the input arguments\n");
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    else if (argc == 2) {
        while ((option = getopt_long(argc, argv, "h", long_opts, &index)) != -1) {
            switch (option) {
                case 'h': {
                    printf("Parameters: --rate <value> --sampling <value> --batch <size> --parallelism <nSource,nMap-Matcher,nSpeed-Calculator,nSink> [--chaining]\n");
                    exit(EXIT_SUCCESS);
                }
            }
        }
    }
    else {
        printf("Error in parsing the input arguments\n");
        exit(EXIT_FAILURE);
    }
    /// data pre-processing
    parse_dataset(file_path);
    create_tuples();
    read_shapefile();

    if (DFF_getMyGroup() == "G0"){
        ff::cout << "Executing TrafficMonitoring with parameters:" << ff::endl;
        if (rate != 0) {
            ff::cout << "  * rate: " << rate << " tuples/second" << ff::endl;
        }
        else {
            ff::cout << "  * rate: full_speed tupes/second" << ff::endl;
        }
        ff::cout << "  * batch size: " << batch_size << ff::endl;
        ff::cout << "  * sampling: " << sampling << ff::endl;
        ff::cout << "  * source: " << source_par_deg << ff::endl;
        ff::cout << "  * map-matcher: " << matcher_par_deg << ff::endl;
        ff::cout << "  * speed-calculator: " << calculator_par_deg << ff::endl;
        ff::cout << "  * sink: " << sink_par_deg << ff::endl;
        ff::cout << "  * topology: source -> map-matcher -> speed-calculator -> sink" << ff::endl;
    }

  
    ff_a2a internalA2A(false, qlen, qlen, true);
    vector<ff_node*> sx;
    vector<ff_node*> dx;
    for(size_t i = 0; i < matcher_par_deg; i++){
        auto* sxp = new ff_pipeline(false, qlen, qlen, true);
        sxp->add_stage(new Source(rate));
        sxp->add_stage(new MapMatcher(road_grid_list));
        sx.push_back(sxp);

        auto* dxp = new ff_pipeline(false, qlen, qlen, true);
        dxp->add_stage(new SpeedCalculator);
        dxp->add_stage(new Sink);

        dx.push_back(dxp);
        internalA2A.createGroup("G"+std::to_string(i)) << sxp << dxp;
    }

    internalA2A.add_firstset(sx, 0, true); // ondemand????
    internalA2A.add_secondset(dx, true);


    /// evaluate topology execution time
    volatile unsigned long start_time_main_usecs = current_time_usecs();
    internalA2A.run_and_wait_end();
    volatile unsigned long end_time_main_usecs = current_time_usecs();
 
    double elapsed_time_seconds = (end_time_main_usecs - start_time_main_usecs) / (1000000.0);
    double throughput = sent_tuples / elapsed_time_seconds;
    ff::cout << "Actual execution time (s): " << elapsed_time_seconds << ff::endl;
    ff::cout << "Tuples sent by source: " << sent_tuples << ff::endl;
    ff::cout << "Measured throughput: " << (int) throughput << " tuples/second" << ff::endl;
    
    return 0;
}