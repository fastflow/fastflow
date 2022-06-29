
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

#include "ogrsf_frmts.h"
#include "ogr_geometry.h"
#include "ogr_feature.h"

using namespace std;

class Polygon {
private:
    vector<OGRPoint*> points;
    double x_min;
    double x_max;
    double y_min;
    double y_max;

    /**
     *  Check if the point p is contained in the polygon.
     *  @param p point
     *  @return true if p is inside the polygon, false otherwise
     */
    bool contains(OGRPoint& p) {
        int counter = 0;

        if (p.getX() < x_min || p.getX() > x_max || p.getY() < y_min || p.getY() > y_max)
            return false;

        for (int i = 0; i < points.size() - 1; i++) {
            if ((points.at(i)->getY() != points.at(i + 1)->getY()) &&
                ((p.getY() < points.at(i)->getY()) || (p.getY() < points.at(i + 1)->getY())) &&
                ((p.getY() > points.at(i)->getY()) || (p.getY() > points.at(i + 1)->getY())))
            {
                double ux = 0;
                double uy = 0;
                double dx = 0;
                double dy = 0;
                int dir = 0;

                if (points.at(i)->getY() > points.at(i + 1)->getY()) {
                    uy = points.at(i)->getY();
                    dy = points.at(i + 1)->getY();
                    ux = points.at(i)->getX();
                    dx = points.at(i + 1)->getX();
                    dir = 0;                        // downward direction
                } else {
                    uy = points.at(i + 1)->getY();
                    dy = points.at(i)->getY();
                    ux = points.at(i + 1)->getX();
                    dx = points.at(i)->getX();
                    dir = 1;                        // upward direction
                }

                double tx = 0;
                if (ux != dx){
                    double k = (uy - dy) / (ux - dx);
                    double b = ((uy - k * ux) + (dy - k * dx)) / 2;
                    tx = (p.getY() - b) / k;
                } else {
                    tx = ux;
                }

                if (tx > p.getX()) {
                    if(dir == 1 && p.getY() != points.at(i + 1)->getY())
                        counter++;
                    else if(p.getY() != points.at(i)->getY())
                        counter++;
                }
            }
        }
        return (counter % 2) != 0;  // (counter % 2) == 0 means point is outside the polygon
    }

    /**
     *  Compute the distance between 2 points (x1, y1) and (x2, y2).
     *  @param p1 x point 1 of coordinates (x1, y1)
     *  @param p2 y point 2 of coordinates (x2, y2)
     *  @return the distance between point 1 and point 2
     */
    double points_distance(OGRPoint& p1, OGRPoint& p2) {
        double x1 = p1.getX();
        double y1 = p1.getY();
        double x2 = p2.getX();
        double y2 = p2.getY();
        return sqrt((x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2));
    }

    /**
     *  Compute the distance between point p0 of coordinates (x0, y0) and the
     *  line where p1 of coordinates (x1, y1) and p2 of coordinates (x2, y2) lay.
     *  @param p1 point of the line
     *  @param p2 point of the line
     *  @param p0 point we want to measure the distance from the line
     *  @return the distance between point p0 and the line
     */
    double point_to_line_distance(OGRPoint& p1, OGRPoint& p2, OGRPoint& p0) {
        double a, b, c; // coefficients of the line ax + by + c = 0
        double distance;

        a = points_distance(p1, p2);
        b = points_distance(p1, p0);
        c = points_distance(p2, p0);

        if (c <= 0.000001 || b <= 0.000001) {       // p0 lays on the line
            distance = 0;
        } else if (a <= 0.000001) {                 // p1 and p2 are equal, the distance is b or c (equal)
            distance = b;
        } else if (c * c >= a * a + b * b) {        // Pythagoras' theorem
            distance = b;
        } else if (b * b >= a * a + c * c) {        // Pythagoras' theorem
            distance = c;
        } else {
            double p = (a + b + c) / 2;
            double s = sqrt(p * (p - a) * (p - b) * (p - c));
            distance = 2 * s / a;
        }
        return distance;
    }

public:

    /**
     *  Constructor.
     *  @param _points vector of points that define the polygon
     */
    Polygon(vector<OGRPoint*>& _points): points(_points),
        x_min(DBL_MAX), x_max(DBL_MIN), y_min(DBL_MAX), y_max(DBL_MIN)
    {
        // update min/max x and min/max y values among all the points
        for (auto& p : points) {
            if (p->getX() < x_min)
                x_min = p->getX();
            if (p->getX() > x_max)
                x_max = p->getX();
            if (p->getY() < y_min)
                y_min = p->getY();
            if (p->getY() > y_max)
                y_max = p->getY();
        }
    }

    /**
     *  Check if the point p lies within the road (the polygon).
     *  @param p point
     *  @param road_width width of the road
     *  @param points points defining the road polygon
     *  @return true if the point falls inside the road polygon area, false otherwise
     */
    bool match_to_road_line(OGRPoint& p, int road_width, double* last_min_dist, int road_id, int* last_road_id) {
        bool match = false;
        for (int i = 0; i < points.size() - 1 && !match; i++) {
            double distance = point_to_line_distance(*points.at(i), *points.at(i + 1), p) * 111.2 * 1000;
            if (distance < *last_min_dist) {
                *last_min_dist = distance;
                *last_road_id = road_id;
            }

            if (distance < road_width / 2.0) // * sqrt(2.0))
                match = true;
        }
        return match;
    }

    /**
     *  Check if the point p lies within the road (the polygon).
     *  @param p point
     *  @param road_width width of the road
     *  @param points points defining the road polygon
     *  @return true if the point falls inside the road polygon area, false otherwise
     */
    bool match_to_road_point(OGRPoint& p, int road_width, double* last_min_dist, int road_id, int* last_road_id) {
        *last_min_dist = DBL_MAX;
        bool match = false;
        for (int i = 0; i < points.size() - 1 && !match; i++) {
            double distance = points_distance(*points.at(i), p);
            if (distance < *last_min_dist) {
                *last_min_dist = distance;
                *last_road_id = road_id;
            }

            if (distance < road_width / 2.0) // * sqrt(2.0))
                match = true;
        }
        return match;
    }

    /**
     *  Destructor.
     */
    ~Polygon() {}
};




class Road_Grid_List {
private:
    unordered_map<string, vector<OGRFeature*>> grid_list;

public:

    /**
     *  @brief Constructor.
     */
    Road_Grid_List() {}

    /**
     *  @brief Method that reads the shapefile.
     *
     *  Read the shapefile, process the road layer and construct a hash map where a key represents the coordinates of a
     *  central point and a value is a list of features corresponding to that point.
     *  @return 0 if the shapefile has been successfully read and the grid_list map created, -1 if an error occurred
     */
    int read_shapefile(const string& shapefile_path) {
        GDALAllRegister();  // registers all format drivers built into GDAL/OGR

        // open the input OGR data source (in this case the shapefile) and use a vector driver
        GDALDataset *dataset = static_cast<GDALDataset*>(GDALOpenEx(shapefile_path.c_str(), GDAL_OF_VECTOR, nullptr, nullptr, nullptr));
        if (dataset == nullptr) {
            ff::cout << "Failed opening GDAL dataset " << shapefile_path << ff::endl;
            return -1;
        }
        //cout << "Successfully opened GDAL dataset " << dataset->GetDescription() << endl;

        // GDALDataset can potentially have many layers associated with it, in this case we have only one layer "roads"
        OGRLayer *roads_layer = dataset->GetLayerByName("roads");
        roads_layer->ResetReading(); // ensure we are starting at the beginning of the "roads" layer

        // feature definition object associated with the layer contains the definitions of all the fields
        OGRFeatureDefn *feature_def = roads_layer->GetLayerDefn();
        OGRFeature *feature = roads_layer->GetNextFeature();

        // iterate through all the features in the "roads layer (return NULL when no more features are available)
        while (feature != nullptr) {
            // extract the geometry from the feature
            OGRGeometry *geometry = feature->GetGeometryRef();
            if (geometry != nullptr) {
                if (geometry->getGeometryType() == 2) {              // GeometryType LINE, GeometryName LINESTRING
                    OGRLineString *line = geometry->toLineString();
                    int length = line->getNumPoints();

                    unique_ptr<OGRPoint> p1(new OGRPoint());
                    unique_ptr<OGRPoint> p2(new OGRPoint());
                    line->getPoint(0, p1.get());
                    line->getPoint(length - 1, p2.get());
                    double center_x = (p1->getX() + p2->getX()) / 2 * 10;
                    double center_y = (p1->getY() + p2->getY()) / 2 * 10;
                    ostringstream map_ID;
                    map_ID << fixed << setprecision(0) << center_y << "_" << fixed << setprecision(0) << center_x;
                    // cout << "Point p1: <" << p1->getX() << ", " << p1->getY() << ">" << endl;
                    // cout << "Point p2: <" << p2->getX() << ", " << p2->getY() << ">" << endl;
                    // cout << "MapID: " << map_ID.str() << endl;

                    if (grid_list.find(map_ID.str()) == grid_list.end())
                        grid_list.emplace(make_pair(map_ID.str(), vector<OGRFeature*>()));
                    else
                        grid_list.at(map_ID.str()).push_back(feature);
                }
            }
            feature = roads_layer->GetNextFeature();
        }
        OGRFeature::DestroyFeature(feature); // method GetNextFeature() returns a copy of the feature that must be freed
        GDALClose(dataset);
        return 0;
    }

    /**
     *  @brief Method that compute a road ID for each GPS position.
     *
     *  Evaluate if there exists a road IDs corresponding to the GPS coordinates contained in point.
     *  @param point GPS coordinates (longitude and latitude) generated by a vehicle
     *  @return the road ID if a match is found, -1 otherwise
     */
    int fetch_road_ID(OGRPoint point) {
        double map_ID_lon = point.getX() * 10;
        double map_ID_lat = point.getY() * 10;
        ostringstream map_ID;
        map_ID << fixed << setprecision(0) << map_ID_lat << "_" << fixed << setprecision(0) << map_ID_lon;
        // cout << "Point point: <" << point.getX() << ", " << point.getY() << ">" << endl;
        // cout << "MapID: " << map_ID.str() << endl;

        int width = 5;
        int last_min_road_ID = -2;
        double last_min_distance = DBL_MAX;
        int grid_count = 0;
        int road_count = 0;

        for (auto entry : grid_list) {
            grid_count++;
            string key = entry.first;
            // cout << "Grid list entry " << grid_count << " key " << key << " vs " << map_ID.str() << endl;

            if (key == map_ID.str()) {
                for (auto feature : entry.second) { // entry.second is a vector<OGRFeature*> (the road_list)
                    road_count++;
                    // retrieve the attribute field road_id of the feature
                    uint64_t road_ID = feature->GetFieldAsInteger64("osm_id");

                    OGRGeometry* geometry = feature->GetGeometryRef();
                    vector<OGRPoint*> points;
                    if (geometry != nullptr) {
                        // wkbFlatten() macro is used to convert the type for a wkbPoint25D (a point with a z coordinate) into
                        // the base 2D geometry type code (wkbPoint); for each 2D geometry type there is a corresponding 2.5D type code;
                        // since the 2D and 2.5D geometry cases are handled by the same C++ class, this code handles 2D or 3D cases properly
                        /*if(wkbFlatten(geometry->getGeometryType()) == wkbPoint) {   // GeometryType POINT
                            cout << "Geometry type POINT" << endl;
                            OGRPoint* p = geometry->toPoint();
                            points.push_back(p);
                            cout << "Point " << p->getX() << ", " << p->getY() << ">" << endl;
                        } else */

                        if (geometry->getGeometryType() == 2) {              // GeometryType LINE, GeometryName LINESTRING
                            OGRLineString* line = geometry->toLineString();
                            for (int i = 0; i < line->getNumPoints(); i++) {
                                OGRPoint* p = new OGRPoint();
                                line->getPoint(i, p);
                                points.push_back(p);
                            }
                        }
                    }

                    Polygon road(points);
                    if (road.match_to_road_line(point, width, &last_min_distance, road_ID, &last_min_road_ID))
                        return road_ID;

                    for (auto p : points)
                        delete p;
                }
                if (last_min_distance < sqrt((width * width) + (10 * 10)))
                    return last_min_road_ID;
                else
                    return -1;
            }

        }
        return -1;
    }

    /**
     *  Destructor.
     */
    ~Road_Grid_List() {}
};