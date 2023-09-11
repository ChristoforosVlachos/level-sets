#include <VVRScene/canvas.h>
#include <VVRScene/mesh.h>
#include <VVRScene/settings.h>
#include <VVRScene/utils.h>
#include <MathGeoLib.h>
#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <set>
#include <chrono>
#include "symmetriceigensolver3x3.h"
#include "canvas.h"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "Eigen/SparseQR"
#include "Eigen/Eigenvalues"
#include "Eigen/SparseCholesky"
#include "math.h"

#define FLAG_SHOW_AXES              1
#define FLAG_SHOW_WIRE              2
#define FLAG_SHOW_SOLID             4
#define FLAG_SHOW_NORMALS           8
#define FLAG_SHOW_AABB             32
#define FLAG_SHOW_ORIGINAL_MODEL  128
#define FLAG_SHOW_TRIANGLES       512
#define FLAG_SHOW_DIFF           1024

// Convenience macros
#define SphereTriangle(s, t) TriangleSphere(t, s)
#define AABBTriangle(a, t) TriangleAABB(t, a)
#define SphereAABB(s, a) AABBSphere(a, s)

typedef struct Interval {
    // contains the minimum and maximum values for the projection on an axis
    float min;
    float max;
} Interval;

class Mesh3DScene : public vvr::Scene
{
public:
    Mesh3DScene();
    const char* getName() const { return "Level Sets"; }
    void keyEvent(unsigned char key, bool up, int modif) override;
    void arrowEvent(vvr::ArrowDir dir, int modif) override;
    void sliderChanged(int slider_id, float val);

    enum _closest_type { FACE, EDGE, VERTEX };
    struct closest_distance {
        vec closest_pt;
        double d_sq;
        int tri_ind;
        enum _closest_type closest_type;
        float incident_angle; // is NULL unless closest_type = VERTEX, used for calculating angle-weighted vertex normals

        bool operator<(const closest_distance& cld) const {
            return d_sq < cld.d_sq;
        }
    };
    struct closest_distance_full {
        vec closest_pt;
        double d_sq;
        std::vector<int> tri_ind;
        enum _closest_type closest_type;
        std::vector<float> incident_angle; // is empty unless closest_type = VERTEX, used for calculating angle-weighted vertex normals

        bool operator<(const closest_distance& cld) const {
            return d_sq < cld.d_sq;
        }
    };
private:
    void draw() override;
    void reset() override;
    void resize() override;
    void Tasks();
    void find_L_Matrix(std::vector<vvr::Triangle>& m_triangles , Eigen::SparseMatrix<float>& L, Eigen::MatrixXf& Coords, Eigen::MatrixXf& DifCoords);
    void Laplacian_Smoothing(vvr::Mesh& m_model, Eigen::MatrixXf Coords, Eigen::MatrixXf DifCoords, float l_coeff);
    void findSDF(vvr::Mesh m_model, int resolution);
    void sampleSpaceAroundModel(vvr::Mesh m_model, int resolution, std::vector<vec>& pts);
    void calculateSDFAtPoint(vvr::Mesh m_model, vec p);
    void doSphereCollision(vvr::Mesh m_model, vvr::Sphere3D s);
    void doAABBCollision(vvr::Mesh m_model, vvr::Box3D aabb);
    void doMeshCollision(vvr::Mesh m1, vvr::Mesh m2);
    void doSDFSphereCollision(vvr::Sphere3D s, const std::vector<float>& distance_field, const std::vector<vec>& pts);
    void doSDFAABBCollision(const vvr::Box3D& aabb, const std::vector<float>& distance_field, const std::vector<vec>& pts);
    void doSDFMeshCollision(const vvr::Mesh& m, const std::vector<float>& distance_field, const std::vector<vec>& pts);
    void doSDFSphereResponse(vvr::Sphere3D s, const std::vector<float>& distance_field, const std::vector<vec>& force_field, const std::vector<vec>& pts, vec& force_vector, vec& force_point_of_application);
    void doSDFAABBResponse(vvr::Box3D aabb, const std::vector<float>& distance_field, const std::vector<vec>& force_field, const std::vector<vec>& pts, vec& force_vector, vec& force_point_of_application);
    void doSDFMeshResponse(const vvr::Mesh& m, const std::vector<float>& distance_field, const std::vector<vec>& force_field, const std::vector<vec>& pts, vec& force_vector, vec& force_point_of_application);
    void exportDistanceAndForceField(const std::vector<float>& distance_field, const std::vector<vec>& force_field, std::string distance_path, std::string force_path);
    void importDistanceAndForceField(const vvr::Mesh& m_model, std::vector<vec>& pts, std::vector<float>& distance_field, std::vector<vec>& force_field, std::string distance_path, std::string force_path);
    void printKeyboardShortcuts();
private:
    int m_style_flag;
    vvr::Colour m_obj_col;
    vvr::Mesh m_model_original, m_model;
    vvr::Sphere3D m_sphere;
    vvr::Box3D m_box;
    vvr::Mesh m_model_original2, m_model2;
    std::vector<vvr::Triangle3D> m_triangles3D;
    std::vector<vvr::Point3D> m_points_diff;
    std::vector<vec> m_points_sample;
    std::vector<vvr::Point3D> m_points_sdf;
    std::vector<vvr::LineSeg3D> m_lines;
    std::vector<vvr::LineSeg3D> m_lines_force;
    Eigen::MatrixXf Coords;
    Eigen::SparseMatrix<float> L;
    Eigen::MatrixXf DifCoords;
    std::vector<float> distance_field;
    float distance_max_p;
    float distance_max_n;
    std::vector<vec> force_field;
    vec m_force_vector;
    vec m_force_point_of_application;
    float m_shrink_coeff;
    float m_inflate_coeff;
    float px;
    float py;
    float pz;
    float size;
    int m_resolution;
    bool m_partb;
    bool m_show_sphere, m_show_box, m_show_mesh2;
    bool m_show_tri, m_show_sdf, m_show_lines;
};
void SparseIdentity(Eigen::SparseMatrix<float>& I, int n);
void SparseDiagonalInverse(Eigen::SparseMatrix<float>& D, Eigen::SparseMatrix<float>& D_inverse, int n);

math::Plane FromTriangle(const vvr::Triangle& t);
float Raycast(const math::Plane& plane, const math::Ray& ray);
float Raycast(const vvr::Triangle& t, const math::Ray& ray);

vec ClosestPoint(const vvr::Triangle& t, const vec& p);
bool TriangleSphere(const vvr::Triangle& t, const vvr::Sphere3D& s);

Interval GetInterval(const vvr::Box3D& aabb, const vec& axis);
Interval GetInterval(const vvr::Triangle& t, const vec& axis);
bool OverlapOnAxis(const vvr::Box3D& aabb, const vvr::Triangle& t, const vec& axis);
bool TriangleAABB(const vvr::Triangle& t, const vvr::Box3D aabb);

bool OverlapOnAxis(const vvr::Triangle& t1, const vvr::Triangle& t2, const vec& axis);
bool TriangleTriangle(const vvr::Triangle& t1, const vvr::Triangle& t2);

vec ClosestPoint(const vvr::Box3D& aabb, const vec& p);
bool AABBSphere(const vvr::Box3D& aabb, const vvr::Sphere3D& s);

bool AABBAABB(const vvr::Box3D& aabb1, const vvr::Box3D& aabb2);

void ExportVector(const std::vector<float>& toExport, std::string path);
void ExportVector(const std::vector<vec>& toExport, std::string path);

void ImportVector(std::vector<float>& toImport, std::string path);
void ImportVector(std::vector<vec>& toImport, std::string path);