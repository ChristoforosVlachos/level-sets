#include "SceneMesh3D.h"

using namespace std;
using namespace vvr;
using namespace Eigen;

Mesh3DScene::Mesh3DScene()
{
    //! Load settings.
    vvr::Shape::DEF_LINE_WIDTH = 4;
    vvr::Shape::DEF_POINT_SIZE = 10;
    m_perspective_proj = true;
    m_hide_log = false;
    m_hide_sliders = false;
    m_bg_col = Colour("768E77");
    m_obj_col = Colour("454545");
    const string objDir = getExePath() + "resources/obj/";
    const string objFile = objDir + "armadillo_low_low.obj";
    m_model_original = vvr::Mesh(objFile);

    const string objFile2 = objDir + "bone.obj";
    m_model_original2 = vvr::Mesh(objFile2);

    reset();
}

void Mesh3DScene::reset()
{
    Scene::reset();
    //! Define what will be vissible by default
    m_style_flag = 0;
    m_style_flag |= FLAG_SHOW_SOLID;
    m_style_flag |= FLAG_SHOW_WIRE;
    m_style_flag |= FLAG_SHOW_AXES;
    m_style_flag |= FLAG_SHOW_AABB;
    m_style_flag |= FLAG_SHOW_ORIGINAL_MODEL;
    m_style_flag |= FLAG_SHOW_TRIANGLES;
    //m_style_flag |= FLAG_SHOW_DIFF;

    m_shrink_coeff = 0.5;
    m_inflate_coeff = -0.5;

    px = py = pz = 5;

    m_model = m_model_original;
    m_model2 = m_model_original2;

    size = 1;

    m_force_vector = vec(0, 0, 0);
    m_force_point_of_application = vec(0, 0, 0);

    m_sphere = Sphere3D(px, py, pz, size, Colour::cyan);

    m_box = Box3D(px - size, py - size, pz - size, px + size, py + size, pz + size, Colour::cyan);

    m_resolution = 5;

    m_partb = false;
    m_show_sphere = m_show_box = m_show_mesh2 = false;
    m_show_tri = m_show_sdf = m_show_lines = true;

    Tasks();
}

void Mesh3DScene::resize()
{
    //! By Making `first_pass` static and initializing it to true,
    //! we make sure that the if block will be executed only once.

    static bool first_pass = true;

    if (first_pass)
    {
        printKeyboardShortcuts();
        m_model_original.setBigSize(getSceneWidth() / 2);
        m_model_original.update();
        m_model = m_model_original;
        m_model_original2.setBigSize(getSceneWidth() / 5);
        m_model_original2.update();
        m_model2 = m_model_original2;
        Tasks();
        first_pass = false;
    }
}

void Mesh3DScene::Tasks()
{
    vector<vvr::Triangle>& m_triangles = m_model.getTriangles();
    vector<vec>& m_vertices = m_model.getVertices();
    int verticesCount = m_vertices.size();

    //cout << "verticesCount = " << verticesCount << endl;


    //convert vec to eigen format
    Coords = MatrixXf(m_model.getVertices().size(), 3);

    for(int i=0;i<m_model.getVertices().size();i++){
        Coords(i, 0) = m_model.getVertices()[i].x;
		Coords(i, 1) = m_model.getVertices()[i].y;
		Coords(i, 2) = m_model.getVertices()[i].z;
    }


    // L is the Laplacian Matrix
    // Coords is a eigen matrix with the original points

    L = SparseMatrix<float>(verticesCount, verticesCount);
    DifCoords = MatrixXf(verticesCount, 3);

    m_points_diff.clear();

    find_L_Matrix(m_triangles, L, Coords, DifCoords);

}

void Mesh3DScene::find_L_Matrix(vector<vvr::Triangle>& m_triangles , SparseMatrix<float>& L, MatrixXf& Coords, MatrixXf& DifCoords) {
    int verticesCount = Coords.rows();
    int trianglesCount = m_triangles.size();

    SparseMatrix<float> A(verticesCount, verticesCount);
    SparseMatrix<float> I(verticesCount, verticesCount);
    SparseMatrix<float> D(verticesCount, verticesCount);
    SparseMatrix<float> D_inverse(verticesCount, verticesCount);


    SparseIdentity(I, verticesCount);

    ///   give values to A and D
    //    A.coeffRef(i, j)  --->> returns a non-const reference to the value of the matrix at position i, j
    //    more here ->  https://eigen.tuxfamily.org/dox/group__TutorialSparse.html


    for (int i = 0; i < trianglesCount; i++) {
        for (int j = 0; j < 2; j++) {
            for (int k = j + 1; k < 3; k++) {
                if (A.coeffRef(m_triangles[i].v[k], m_triangles[i].v[j]) == 0) {
                    A.coeffRef(m_triangles[i].v[k], m_triangles[i].v[j]) = 1;
                    A.coeffRef(m_triangles[i].v[j], m_triangles[i].v[k]) = 1;
                    D.coeffRef(m_triangles[i].v[j], m_triangles[i].v[j])++;
                    D.coeffRef(m_triangles[i].v[k], m_triangles[i].v[k])++;
                }
            }
        }
    }

    ///  The inverse of matrix D will also be a diagonal n×n matrix in the following form:
    //             | d1  0    0|
    //        D =  | 0   d2   0|
    //             | 0   0   d3|
    //             | 1/d1  0     0|
    // D_inverse=  |  0   1/d2   0|
    //             |  0    0  1/d3|

    SparseDiagonalInverse(D, D_inverse, verticesCount);

    ///  find L from L= I - D_inverse * A
    L = I - D_inverse * A;

    DifCoords = L * Coords;


    vector<float> weights;
    float max_weight = -1, min_weight = 1e9;
    for (int i = 0; i < verticesCount; i++) {
        float weight = SQUARE(DifCoords(i, 0)) + SQUARE(DifCoords(i, 1)) + SQUARE(DifCoords(i, 2));
        weights.push_back(weight);
        
        if (weight > max_weight) max_weight = weight;
        if (weight < min_weight) min_weight = weight;
    }

    for (int i = 0; i < verticesCount; i++) {
        // linear
        //m_points_diff.push_back(Point3D(Coords(i, 0), Coords(i, 1), Coords(i, 2), Colour(255 * (weights[i] - min_weight) / (max_weight - min_weight), 255 * (max_weight - weights[i]) / (max_weight - min_weight), 0)));

        // logarithmic
        m_points_diff.push_back(Point3D(Coords(i, 0), Coords(i, 1), Coords(i, 2), Colour(255 * log10(weights[i] - min_weight) / log10(max_weight - min_weight), 255 * log10(max_weight - weights[i]) / log10(max_weight - min_weight), 0)));
    }
}

void Mesh3DScene::Laplacian_Smoothing(Mesh& m_model, MatrixXf Coords, MatrixXf DifCoords, float l_coeff)
{
    vector<vec>& m_model_vertices = m_model.getVertices();

    for (int i = 0; i < m_model_vertices.size(); i++) {
        m_model_vertices[i].x = m_model_vertices[i].x - l_coeff * DifCoords(i, 0);
        m_model_vertices[i].y = m_model_vertices[i].y - l_coeff * DifCoords(i, 1);
        m_model_vertices[i].z = m_model_vertices[i].z - l_coeff * DifCoords(i, 2);
    }
}

void Mesh3DScene::findSDF(Mesh m_model, int resolution)
{
    m_triangles3D.clear();
    m_points_sample.clear();
    sampleSpaceAroundModel(m_model, resolution, m_points_sample);

    m_lines.clear();

    distance_field.clear();
    force_field.clear();

    distance_max_p = 0;
    distance_max_n = 0;

    for (vec& p : m_points_sample) {
        calculateSDFAtPoint(m_model, p);
    }
    //for_each(m_points_sample.begin(), m_points_sample.end(), [&](vec p) {calculateSDFAtPoint(m_model, p); });
}

void Mesh3DScene::sampleSpaceAroundModel(Mesh m_model, int resolution, vector<vec>& pts)
{
    AABB aabb = m_model.getAABB();

    float minx = aabb.MinX();
    float miny = aabb.MinY();
    float minz = aabb.MinZ();
    float maxx = aabb.MaxX();
    float maxy = aabb.MaxY();
    float maxz = aabb.MaxZ();

    float Lx = maxx - minx;
    float Ly = maxy - miny;
    float Lz = maxz - minz;

    float margin = 0.1;

    minx -= margin * Lx;
    miny -= margin * Ly;
    minz -= margin * Lz;
    maxx += margin * Lx;
    maxy += margin * Ly;
    maxz += margin * Lz;

    float stepx = (maxx - minx) / (resolution - 1);
    float stepy = (maxy - miny) / (resolution - 1);
    float stepz = (maxz - minz) / (resolution - 1);

    pts.clear();

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                pts.push_back(vec(minx + i * stepx, miny + j * stepy, minz + k * stepz));
            }
        }
    }

}

void Mesh3DScene::calculateSDFAtPoint(Mesh m_model, vec p)
{
    vector<vvr::Triangle> m_triangles = m_model.getTriangles();

    vector<closest_distance> closest;

    for (int i = 0; i < m_triangles.size(); i++) {
        vvr::Triangle& tri = m_triangles[i];
        // https://stackoverflow.com/a/9605695
        vec n = tri.getNormal();
        vec orig = tri.v1();
        vec v = p - orig;
        double dist = Dot(v, n);
        vec pp = p - dist * n;

        // https://math.stackexchange.com/a/28552
        double area = math::Cross(tri.v2() - tri.v1(), tri.v3() - tri.v1()).Length() / 2.0;
        double a = math::Cross(tri.v2() - pp, tri.v3() - pp).Length() / (2.0 * area);
        double b = math::Cross(tri.v3() - pp, tri.v1() - pp).Length() / (2.0 * area);
        double c = math::Cross(tri.v1() - pp, tri.v2() - pp).Length() / (2.0 * area);

        if (a > 0 && a < 1 && b > 0 && b < 1 && c > 0 && c < 1 && a + b + c > 0.99999 && a + b + c < 1.00001) {
            //cout << "inside!" << endl;
            //m_lines.push_back(LineSeg3D(p.x, p.y, p.z, pp.x, pp.y, pp.z, Colour::magenta));
            closest.push_back({ pp, p.DistanceSq(pp), i, FACE, NULL });
        }
        else {
            double d_v1 = (pp - tri.v1()).LengthSq();
            double d_v2 = (pp - tri.v2()).LengthSq();
            double d_v3 = (pp - tri.v3()).LengthSq();
            double d_v1v2 = (pp - tri.v1()).LengthSq() - SQUARE(Dot((tri.v2() - tri.v1()).Normalized(), pp - tri.v1()));
            double d_v2v3 = (pp - tri.v2()).LengthSq() - SQUARE(Dot((tri.v3() - tri.v2()).Normalized(), pp - tri.v2()));
            double d_v3v1 = (pp - tri.v3()).LengthSq() - SQUARE(Dot((tri.v1() - tri.v3()).Normalized(), pp - tri.v3()));

            // assign a penalty to distances not actually on the triangle
            if (Dot((tri.v2() - tri.v1()).Normalized(), pp - tri.v1()) < 0 ||
                SQUARE(Dot((tri.v2() - tri.v1()).Normalized(), pp - tri.v1())) > (tri.v2() - tri.v1()).LengthSq()) {
                d_v1v2 = 1e10;
            }
            if (Dot((tri.v3() - tri.v2()).Normalized(), pp - tri.v2()) < 0 ||
                SQUARE(Dot((tri.v3() - tri.v2()).Normalized(), pp - tri.v2())) > (tri.v3() - tri.v2()).LengthSq()) {
                d_v2v3 = 1e10;
            }
            if (Dot((tri.v1() - tri.v3()).Normalized(), pp - tri.v3()) < 0 ||
                SQUARE(Dot((tri.v1() - tri.v3()).Normalized(), pp - tri.v3())) > (tri.v1() - tri.v3()).LengthSq()) {
                d_v3v1 = 1e10;
            }
            
            if (d_v1 <= d_v2 && d_v1 <= d_v3 && d_v1 <= d_v1v2 && d_v1 <= d_v2v3 && d_v1 <= d_v3v1) {
                //m_lines.push_back(LineSeg3D(p.x, p.y, p.z, tri.v1().x, tri.v1().y, tri.v1().z, Colour::magenta));
                closest.push_back({ tri.v1(), p.DistanceSq(tri.v1()), i, VERTEX, (tri.v2() - tri.v1()).AngleBetween(tri.v3() - tri.v1()) });
            }
            if (d_v2 <= d_v1 && d_v2 <= d_v3 && d_v2 <= d_v1v2 && d_v2 <= d_v2v3 && d_v2 <= d_v3v1) {
                //m_lines.push_back(LineSeg3D(p.x, p.y, p.z, tri.v2().x, tri.v2().y, tri.v2().z, Colour::magenta));
                closest.push_back({ tri.v2(), p.DistanceSq(tri.v2()), i, VERTEX, (tri.v3() - tri.v2()).AngleBetween(tri.v1() - tri.v2()) });
            }
            if (d_v3 <= d_v1 && d_v3 <= d_v2 && d_v3 <= d_v1v2 && d_v3 <= d_v2v3 && d_v3 <= d_v3v1) {
                //m_lines.push_back(LineSeg3D(p.x, p.y, p.z, tri.v3().x, tri.v3().y, tri.v3().z, Colour::magenta));
                closest.push_back({ tri.v3(), p.DistanceSq(tri.v3()), i, VERTEX, (tri.v1() - tri.v3()).AngleBetween(tri.v2() - tri.v3()) });
            }
            if (d_v1v2 <= d_v1 && d_v1v2 <= d_v2 && d_v1v2 <= d_v3 && d_v1v2 <= d_v2v3 && d_v1v2 <= d_v3v1) {
                vec v1v2 = tri.v1() + Dot((tri.v2() - tri.v1()).Normalized(), pp - tri.v1()) * (tri.v2() - tri.v1()).Normalized();
                //m_lines.push_back(LineSeg3D(p.x, p.y, p.z, v1v2.x, v1v2.y, v1v2.z, Colour::magenta));
                closest.push_back({ v1v2, p.DistanceSq(v1v2), i, EDGE, NULL });
            }
            if (d_v2v3 <= d_v1 && d_v2v3 <= d_v2 && d_v2v3 <= d_v3 && d_v2v3 <= d_v1v2 && d_v2v3 <= d_v3v1) {
                vec v2v3 = tri.v2() + Dot((tri.v3() - tri.v2()).Normalized(), pp - tri.v2()) * (tri.v3() - tri.v2()).Normalized();
                //m_lines.push_back(LineSeg3D(p.x, p.y, p.z, v2v3.x, v2v3.y, v2v3.z, Colour::magenta));
                closest.push_back({ v2v3, p.DistanceSq(v2v3), i, EDGE, NULL });
            }
            if (d_v3v1 <= d_v1 && d_v3v1 <= d_v2 && d_v3v1 <= d_v3 && d_v3v1 <= d_v1v2 && d_v3v1 <= d_v2v3) {
                vec v3v1 = tri.v3() + Dot((tri.v1() - tri.v3()).Normalized(), pp - tri.v3()) * (tri.v1() - tri.v3()).Normalized();
                //m_lines.push_back(LineSeg3D(p.x, p.y, p.z, v3v1.x, v3v1.y, v3v1.z, Colour::magenta));
                closest.push_back({ v3v1, p.DistanceSq(v3v1), i, EDGE, NULL });
            }
        }

    }

    std::sort(closest.begin(), closest.end());

    closest_distance_full closest_full = {};
    closest_full.closest_pt = closest[0].closest_pt;
    closest_full.d_sq = closest[0].d_sq;
    closest_full.tri_ind.push_back(closest[0].tri_ind);
    closest_full.closest_type = closest[0].closest_type;
    if (closest_full.closest_type == VERTEX) {
        closest_full.incident_angle.push_back(closest[0].incident_angle);
    }

    int ind = 1;
    while (closest[ind].d_sq - closest_full.d_sq > -0.001 && closest[ind].d_sq - closest_full.d_sq < 0.001) {
        if (closest[ind].closest_type != closest_full.closest_type) { // may not be neccessary for smaller precision bias (line above)
            //cout << "skipping" << endl;
        }
        else {
            closest_full.tri_ind.push_back(closest[ind].tri_ind);
            if (closest_full.closest_type == VERTEX) {
                closest_full.incident_angle.push_back(closest[ind].incident_angle);
            }
        }
        ind++;
    }

    m_lines.push_back(LineSeg3D(p.x, p.y, p.z, closest_full.closest_pt.x, closest_full.closest_pt.y, closest_full.closest_pt.z, Colour::magenta));

    for (int index : closest_full.tri_ind) {
        m_triangles3D.push_back(Triangle3D(m_triangles[index].v1().x, m_triangles[index].v1().y, m_triangles[index].v1().z, m_triangles[index].v2().x, m_triangles[index].v2().y, m_triangles[index].v2().z, m_triangles[index].v3().x, m_triangles[index].v3().y, m_triangles[index].v3().z, Colour::blue));
    }

    if (closest_full.closest_type == FACE) {
        float product = Dot(m_triangles[closest_full.tri_ind[0]].getNormal(), (p - closest_full.closest_pt));
        int sign = (product <= 0) ? 1 : -1;

        float d = Sqrt(closest_full.d_sq);
        if (sign == 1) {
            // distance field
            distance_field.push_back(sign * d);
            if (d > distance_max_p) distance_max_p = d;

            // force field
            force_field.push_back(vec(0, 0, 0));
        }
        else { // sign == -1
            // distance field
            distance_field.push_back(sign * d);
            if (d > distance_max_n) distance_max_n = d;

            // force field
            force_field.push_back(-d * m_triangles[closest_full.tri_ind[0]].getNormal());
        }
    }
    else if (closest_full.closest_type == EDGE) {
        vec avg_normal = vec(0, 0, 0);
        for (int i = 0; i < closest_full.tri_ind.size(); i++) {
            avg_normal += m_triangles[closest_full.tri_ind[i]].getNormal();
        }
        avg_normal /= closest_full.tri_ind.size();

        float product = Dot(avg_normal, (p - closest_full.closest_pt));
        int sign = (product <= 0) ? 1 : -1;

        float d = Sqrt(closest_full.d_sq);
        if (sign == 1) {
            // distance field
            distance_field.push_back(sign * d);
            if (d > distance_max_p) distance_max_p = d;

            // force field
            force_field.push_back(vec(0, 0, 0));
        }
        else { // sign == -1
            // distance field
            distance_field.push_back(sign * d);
            if (d > distance_max_n) distance_max_n = d;

            // force field
            force_field.push_back(-d * avg_normal);
        }
    }
    else { // closest_full.closest_type == VERTEX
        vec avg_normal = vec(0, 0, 0);
        float weight_sum = 0;
        for (int i = 0; i < closest_full.tri_ind.size(); i++) {
            avg_normal += m_triangles[closest_full.tri_ind[i]].getNormal() * closest_full.incident_angle[i];
            weight_sum += closest_full.incident_angle[i];
        }
        avg_normal /= weight_sum;

        float product = Dot(avg_normal, (p - closest_full.closest_pt));
        int sign = (product <= 0) ? 1 : -1;

        float d = Sqrt(closest_full.d_sq);
        if (sign == 1) {
            // distance field
            distance_field.push_back(sign * d);
            if (d > distance_max_p) distance_max_p = d;

            // force field
            force_field.push_back(vec(0, 0, 0));
        }
        else { // sign == -1
            // distance field
            distance_field.push_back(sign * d);
            if (d > distance_max_n) distance_max_n = d;

            // force field
            force_field.push_back(-d * avg_normal);
        }
    }
}

void Mesh3DScene::doSphereCollision(vvr::Mesh m_model, vvr::Sphere3D s)
{
    auto clock_start = std::chrono::high_resolution_clock::now();
    vector<vvr::Triangle> m_triangles = m_model.getTriangles();

    m_triangles3D.clear();
    for (vvr::Triangle& tri : m_triangles) {
        if (TriangleSphere(tri, s)) {
            cout << "Collision! (intersection)" << endl;
            auto clock_end = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
            cout << "Took " << dt << " milliseconds." << endl;
            return;
        }
    }

    int ray_intersections = 0;
    for (vvr::Triangle& tri : m_triangles) {
        if (Raycast(tri, Ray(vec(s.x, s.y, s.z), vec(1, 0, 0))) >= 0) {
            ray_intersections++;
            m_triangles3D.push_back(Triangle3D(tri.v1().x, tri.v1().y, tri.v1().z, tri.v2().x, tri.v2().y, tri.v2().z, tri.v3().x, tri.v3().y, tri.v3().z, Colour::green));
        }
    }

    if (ray_intersections % 2 == 0) {
        cout << "No collision!" << endl;
        //cout << ray_intersections << endl;
    }
    else {
        cout << "Collision! (fully submerged)" << endl;
        //cout << ray_intersections << endl;
    }

    auto clock_end = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
    cout << "Took " << dt << " milliseconds." << endl;
}

void Mesh3DScene::doAABBCollision(vvr::Mesh m_model, vvr::Box3D aabb)
{
    auto clock_start = std::chrono::high_resolution_clock::now();
    vector<vvr::Triangle> m_triangles = m_model.getTriangles();

    m_triangles3D.clear();
    for (vvr::Triangle& tri : m_triangles) {
        if (TriangleAABB(tri, aabb)) {
            cout << "Collision! (intersection)" << endl;
            auto clock_end = std::chrono::high_resolution_clock::now();
            double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
            cout << "Took " << dt << " milliseconds." << endl;
            return;
        }
    }

    int ray_intersections = 0;
    for (vvr::Triangle& tri : m_triangles) {
        if (Raycast(tri, Ray(vec(px, py, pz), vec(1, 0, 0))) >= 0) {
            ray_intersections++;
            m_triangles3D.push_back(Triangle3D(tri.v1().x, tri.v1().y, tri.v1().z, tri.v2().x, tri.v2().y, tri.v2().z, tri.v3().x, tri.v3().y, tri.v3().z, Colour::green));
        }
    }

    if (ray_intersections % 2 == 0) {
        cout << "No collision!" << endl;
        //cout << ray_intersections << endl;
    }
    else {
        cout << "Collision! (fully submerged)" << endl;
        //cout << ray_intersections << endl;
    }

    auto clock_end = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
    cout << "Took " << dt << " milliseconds." << endl;
}

void Mesh3DScene::doMeshCollision(vvr::Mesh m1, vvr::Mesh m2)
{
    auto clock_start = std::chrono::high_resolution_clock::now();
    vector<vvr::Triangle> m_triangles1 = m1.getTriangles();
    vector<vvr::Triangle> m_triangles2 = m2.getTriangles();

    m_triangles3D.clear();
    for (vvr::Triangle& t1 : m_triangles1) {
        for (vvr::Triangle& t2 : m_triangles2) {
            if (TriangleTriangle(t1, t2)) {
                cout << "Collision! (intersection)" << endl;
                auto clock_end = std::chrono::high_resolution_clock::now();
                double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
                cout << "Took " << dt << " milliseconds." << endl;
                return;
            }
        }
    }

    int ray_intersections = 0;
    vec some_vertex = m2.getVertices()[0];
    for (vvr::Triangle& tri : m_triangles1) {
        if (Raycast(tri, Ray(some_vertex, vec(1, 0, 0))) >= 0) {
            ray_intersections++;
            m_triangles3D.push_back(Triangle3D(tri.v1().x, tri.v1().y, tri.v1().z, tri.v2().x, tri.v2().y, tri.v2().z, tri.v3().x, tri.v3().y, tri.v3().z, Colour::green));
        }
    }

    if (ray_intersections % 2 == 0) {
        cout << "No collision!" << endl;
        //cout << ray_intersections << endl;
    }
    else {
        cout << "Collision! (fully submerged)" << endl;
        //cout << ray_intersections << endl;
    }

    auto clock_end = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
    cout << "Took " << dt << " milliseconds." << endl;
}

void Mesh3DScene::doSDFSphereCollision(vvr::Sphere3D s, const std::vector<float>& distance_field, const std::vector<vec>& pts)
{
    if (distance_field.empty()) {
        cout << "Distance field not generated! Please, generate or load one." << endl;
        return;
    }
    auto clock_start = std::chrono::high_resolution_clock::now();
    int resolution = cbrt(pts.size());
    vec difference = pts[SQUARE(resolution) + resolution + 1] - pts[0];
    float lx = difference.x / 2.0;
    float ly = difference.y / 2.0;
    float lz = difference.z / 2.0;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                if (distance_field[i * SQUARE(resolution) + j * resolution + k] > 0) {
                    continue;
                }
                vec p = pts[i * SQUARE(resolution) + j * resolution + k];
                Box3D aabb = Box3D(p.x - lx, p.y - lx, p.z - lx, p.x + lx, p.y + lx, p.z + lx, Colour::red);
                if (AABBSphere(aabb, s)) {
                    cout << "Collision!" << endl;
                    auto clock_end = std::chrono::high_resolution_clock::now();
                    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
                    cout << "Took " << dt << " milliseconds." << endl;
                    return;
                }
            }
        }
    }
    cout << "No collision!" << endl;
    auto clock_end = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
    cout << "Took " << dt << " milliseconds." << endl;
}

void Mesh3DScene::doSDFAABBCollision(const vvr::Box3D& aabb, const std::vector<float>& distance_field, const std::vector<vec>& pts)
{
    if (distance_field.empty()) {
        cout << "Distance field not generated! Please, generate or load one." << endl;
        return;
    }
    auto clock_start = std::chrono::high_resolution_clock::now();
    int resolution = cbrt(pts.size());
    vec difference = pts[SQUARE(resolution) + resolution + 1] - pts[0];
    float lx = difference.x / 2.0;
    float ly = difference.y / 2.0;
    float lz = difference.z / 2.0;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                if (distance_field[i * SQUARE(resolution) + j * resolution + k] > 0) {
                    continue;
                }
                vec p = pts[i * SQUARE(resolution) + j * resolution + k];
                Box3D aabb2 = Box3D(p.x - lx, p.y - lx, p.z - lx, p.x + lx, p.y + lx, p.z + lx, Colour::red);
                if (AABBAABB(aabb, aabb2)) {
                    cout << "Collision!" << endl;
                    auto clock_end = std::chrono::high_resolution_clock::now();
                    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
                    cout << "Took " << dt << " milliseconds." << endl;
                    return;
                }
            }
        }
    }
    cout << "No collision!" << endl;
    auto clock_end = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
    cout << "Took " << dt << " milliseconds." << endl;
}

void Mesh3DScene::doSDFMeshCollision(const vvr::Mesh& m, const std::vector<float>& distance_field, const std::vector<vec>& pts)
{
    if (distance_field.empty()) {
        cout << "Distance field not generated! Please, generate or load one." << endl;
        return;
    }
    auto clock_start = std::chrono::high_resolution_clock::now();
    const vector<vvr::Triangle>& m_triangles = m.getTriangles();

    int resolution = cbrt(pts.size());
    vec difference = pts[SQUARE(resolution) + resolution + 1] - pts[0];
    float lx = difference.x / 2.0;
    float ly = difference.y / 2.0;
    float lz = difference.z / 2.0;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                if (distance_field[i * SQUARE(resolution) + j * resolution + k] > 0) {
                    continue;
                }
                vec p = pts[i * SQUARE(resolution) + j * resolution + k];
                Box3D aabb = Box3D(p.x - lx, p.y - lx, p.z - lx, p.x + lx, p.y + lx, p.z + lx, Colour::red);
                for (const vvr::Triangle& tri : m_triangles) {
                    if (AABBTriangle(aabb, tri)) {
                        cout << "Collision!" << endl;
                        auto clock_end = std::chrono::high_resolution_clock::now();
                        double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
                        cout << "Took " << dt << " milliseconds." << endl;
                        return;
                    }
                }
            }
        }
    }
    cout << "No collision!" << endl;
    auto clock_end = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration_cast<std::chrono::milliseconds>(clock_end - clock_start).count();
    cout << "Took " << dt << " milliseconds." << endl;
}

void Mesh3DScene::doSDFSphereResponse(vvr::Sphere3D s, const std::vector<float>& distance_field, const std::vector<vec>& force_field, const std::vector<vec>& pts, vec& force_vector, vec& force_point_of_application)
{
    if (distance_field.empty()) {
        cout << "Distance field not generated! Please, generate or load one." << endl;
        return;
    }
    int resolution = cbrt(pts.size());
    vec difference = pts[SQUARE(resolution) + resolution + 1] - pts[0];
    float lx = difference.x / 2.0;
    float ly = difference.y / 2.0;
    float lz = difference.z / 2.0;

    vec average_vec = vec(0, 0, 0);
    int average_count = 0;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                int index = i * SQUARE(resolution) + j * resolution + k;
                if (distance_field[index] > 0) {
                    continue;
                }
                vec p = pts[index];
                Box3D aabb = Box3D(p.x - lx, p.y - lx, p.z - lx, p.x + lx, p.y + lx, p.z + lx, Colour::red);
                if (AABBSphere(aabb, s)) {
                    average_vec += force_field[index];
                    average_count++;
                    //cout << "Collision " << average_count << endl;
                }
            }
        }
    }
    if (average_count > 0) {
        average_vec /= static_cast<float>(average_count);
        force_vector = 10 * average_vec;
        force_point_of_application = vec(s.x, s.y, s.z);
        cout << "Collision!" << endl;
    }
    else {
        cout << "No collision!" << endl;
    }
}

void Mesh3DScene::doSDFAABBResponse(vvr::Box3D aabb, const std::vector<float>& distance_field, const std::vector<vec>& force_field, const std::vector<vec>& pts, vec& force_vector, vec& force_point_of_application)
{
    if (distance_field.empty()) {
        cout << "Distance field not generated! Please, generate or load one." << endl;
        return;
    }
    int resolution = cbrt(pts.size());
    vec difference = pts[SQUARE(resolution) + resolution + 1] - pts[0];
    float lx = difference.x / 2.0;
    float ly = difference.y / 2.0;
    float lz = difference.z / 2.0;

    vec average_vec = vec(0, 0, 0);
    int average_count = 0;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                int index = i * SQUARE(resolution) + j * resolution + k;
                if (distance_field[index] > 0) {
                    continue;
                }
                vec p = pts[index];
                Box3D aabb2 = Box3D(p.x - lx, p.y - lx, p.z - lx, p.x + lx, p.y + lx, p.z + lx, Colour::red);
                if (AABBAABB(aabb, aabb2)) {
                    average_vec += force_field[index];
                    average_count++;
                    //cout << "Collision " << average_count << endl;
                }
            }
        }
    }
    if (average_count > 0) {
        average_vec /= static_cast<float>(average_count);
        force_vector = 10 * average_vec;
        force_point_of_application = vec((aabb.x1 + aabb.x2) / 2.0, (aabb.y1 + aabb.y2) / 2.0, (aabb.z1 + aabb.z2) / 2.0);
        cout << "Collision!" << endl;
    }
    else {
        cout << "No collision!" << endl;
    }
}

void Mesh3DScene::doSDFMeshResponse(const vvr::Mesh& m, const std::vector<float>& distance_field, const std::vector<vec>& force_field, const std::vector<vec>& pts, vec& force_vector, vec& force_point_of_application)
{
    if (distance_field.empty()) {
        cout << "Distance field not generated! Please, generate or load one." << endl;
        return;
    }
    const vector<vvr::Triangle>& m_triangles = m.getTriangles();

    int resolution = cbrt(pts.size());
    vec difference = pts[SQUARE(resolution) + resolution + 1] - pts[0];
    float lx = difference.x / 2.0;
    float ly = difference.y / 2.0;
    float lz = difference.z / 2.0;

    vec average_vec = vec(0, 0, 0);
    vec average_poa = vec(0, 0, 0);
    int average_count = 0;

    for (int i = 0; i < resolution; i++) {
        for (int j = 0; j < resolution; j++) {
            for (int k = 0; k < resolution; k++) {
                int index = i * SQUARE(resolution) + j * resolution + k;
                if (distance_field[index] > 0) {
                    continue;
                }
                vec p = pts[index];
                Box3D aabb = Box3D(p.x - lx, p.y - lx, p.z - lx, p.x + lx, p.y + lx, p.z + lx, Colour::red);
                for (const vvr::Triangle& tri : m_triangles) {
                    if (AABBTriangle(aabb, tri)) {
                        average_vec += force_field[index];
                        average_poa += tri.getCenter();
                        average_count++;
                        //cout << "Collision " << average_count << endl;
                    }
                }
            }
        }
    }
    if (average_count > 0) {
        average_vec /= static_cast<float>(average_count);
        average_poa /= static_cast<float>(average_count);
        force_vector = 10 * average_vec;
        force_point_of_application = average_poa;
        cout << "Collision!" << endl;
    }
    else {
        cout << "No collision!" << endl;
    }
}

void Mesh3DScene::exportDistanceAndForceField(const std::vector<float>& distance_field, const std::vector<vec>& force_field, string distance_path, string force_path)
{
    ExportVector(distance_field, distance_path);
    ExportVector(force_field, force_path);
}

void Mesh3DScene::importDistanceAndForceField(const vvr::Mesh& m_model, std::vector<vec>& pts, std::vector<float>& distance_field, std::vector<vec>& force_field, string distance_path, string force_path)
{
    ImportVector(distance_field, distance_path);
    auto it = minmax_element(distance_field.begin(), distance_field.end());
    distance_max_n = -*it.first;
    distance_max_p = *it.second;
    ImportVector(force_field, force_path);
    sampleSpaceAroundModel(m_model, cbrt(distance_field.size()), pts);
}

void Mesh3DScene::arrowEvent(ArrowDir dir, int modif)
{

}

void Mesh3DScene::keyEvent(unsigned char key, bool up, int modif)
{
    Scene::keyEvent(key, up, modif);
    key = tolower(key);

    switch (key)
    {
        case 's': m_style_flag ^= FLAG_SHOW_SOLID; break;
        case 'w': m_style_flag ^= FLAG_SHOW_WIRE; break;
        case 'n': m_style_flag ^= FLAG_SHOW_NORMALS; break;
        case 'a': m_style_flag ^= FLAG_SHOW_AXES; break;
        case 'b': m_style_flag ^= FLAG_SHOW_AABB; break;
        case 'd': m_style_flag ^= FLAG_SHOW_DIFF; break;

        case '-': cout << "Shrinking... "; Laplacian_Smoothing(m_model, Coords, DifCoords, m_shrink_coeff); Tasks(); cout << "Done" << endl; break;
        case '+': cout << "Inflating... "; Laplacian_Smoothing(m_model, Coords, DifCoords, m_inflate_coeff); Tasks(); cout << "Done" << endl; break;
        case '*': cout << "Inflating... "; Laplacian_Smoothing(m_model, Coords, DifCoords, m_inflate_coeff); Tasks(); cout << "Smoothing... "; Laplacian_Smoothing(m_model, Coords, DifCoords, -0.75 * m_inflate_coeff); Tasks(); cout << "Done" << endl; break;

        case ' ':
            if (m_partb)
                if (m_show_sphere && !shiftDown(modif)) {
                    doSphereCollision(m_model, m_sphere);
                }
                else if (m_show_box && !shiftDown(modif)) {
                    doAABBCollision(m_model, m_box);
                }
                else if (m_show_mesh2 && !shiftDown(modif)) {
                    doMeshCollision(m_model, m_model2);
                }
                else if (m_show_sphere && shiftDown(modif)) {
                    doSDFSphereCollision(m_sphere, distance_field, m_points_sample);
                }
                else if (m_show_box && shiftDown(modif)) {
                    doSDFAABBCollision(m_box, distance_field, m_points_sample);
                }
                else if (m_show_mesh2 && shiftDown(modif)) {
                    doSDFMeshCollision(m_model2, distance_field, m_points_sample);
                }
            break;
        //case '\\': m_points_sample.clear(); sampleSpaceAroundModel(m_model, m_resolution, m_points_sample); break;
        case ';': findSDF(m_model, m_resolution); break;
        case '\'': m_show_sdf = !m_show_sdf; break;

        case '0': m_partb = !m_partb; printKeyboardShortcuts(); break;
        case 'p':
            if (m_partb)
                if (m_show_sphere) {
                    m_show_sphere = false;
                }
                else {
                    m_show_sphere = true;
                    m_show_box = false;
                    m_show_mesh2 = false;
                }
            break;
        case 'x':
            if (m_partb)
                if (m_show_box) {
                    m_show_box = false;
                }
                else {
                    m_show_box = true;
                    m_show_sphere = false;
                    m_show_mesh2 = false;
                }
            break;
        case 'm':
            if (m_partb)
                if (m_show_mesh2) {
                    m_show_mesh2 = false;
                }
                else {
                    m_show_mesh2 = true;
                    m_show_sphere = false;
                    m_show_box = false;
                }
            break;
        case 'e':
            if (m_partb)
                if (m_show_sphere) {
                    doSDFSphereResponse(m_sphere, distance_field, force_field, m_points_sample, m_force_vector, m_force_point_of_application);
                }
                else if (m_show_box) {
                    doSDFAABBResponse(m_box, distance_field, force_field, m_points_sample, m_force_vector, m_force_point_of_application);
                }
                else if (m_show_mesh2) {
                    doSDFMeshResponse(m_model2, distance_field, force_field, m_points_sample, m_force_vector, m_force_point_of_application);
                }
            break;
        
        case 'c': exportDistanceAndForceField(distance_field, force_field, getExePath() + "data/distance.out", getExePath() + "data/force.out"); break;
        case 'z': importDistanceAndForceField(m_model, m_points_sample, distance_field, force_field, getExePath() + "data/distance.out", getExePath() + "data/force.out"); break;

        case 't': m_show_tri = !m_show_tri; break;
        case 'l': m_show_lines = !m_show_lines; break;

        case '?': printKeyboardShortcuts(); break;
    }
}

void Mesh3DScene::printKeyboardShortcuts()
{
    if (!m_partb) {
        std::cout << "Keyboard shortcuts:"
            << std::endl << "? => This shortcut list:"
            << std::endl << "0 => Switch to Part B"
            << std::endl << "D => Visualize differential coordinates"
            << std::endl << "- => Shrink"
            << std::endl << "+ => Inflate"
            << std::endl << "* => Smart inflate"
            << std::endl << "Smart inflation performs 75% smoothing after inflating. It is advised that you increase the inflation coefficient."
            << std::endl
            << std::endl << "Sliders:"
            << std::endl << "S0 => shrinking coefficient"
            << std::endl << "S1 => inflation coefficient"
            << std::endl << std::endl;
    }
    else {
        std::cout << "Keyboard shortcuts:"
            << std::endl << "? => This shortcut list:"
            << std::endl << "0 => Switch to Part A"
            << std::endl << "; => Generate SDF and force field"
            << std::endl << "C => Export SDF and force field"
            << std::endl << "Z => Import SDF and force field"
            << std::endl << "' => Display SDF and force field"
            << std::endl << "P => Show sphere"
            << std::endl << "X => Show axis aligned box"
            << std::endl << "M => Show 2nd mesh"
            << std::endl << "SPACE => Perform collision detection using only the meshes"
            << std::endl << "SHIFT + SPACE => Perform collision detection using the SDF"
            << std::endl << "E => Calculate collision response force"
            << std::endl << "T => Show/Hide auxiliary triangles"
            << std::endl << "L => Show/Hide auxiliary lines"
            << std::endl
            << std::endl << "Sliders:"
            << std::endl << "S0 => probe point x coordinate"
            << std::endl << "S1 => probe point y coordinate"
            << std::endl << "S2 => probe point z coordinate"
            << std::endl << "S3 => sphere/box/mesh2 size"
            << std::endl << "S4 => SDF resolution (DON'T SET TOO LARGE)"
            << std::endl << std::endl;
    }
}

void Mesh3DScene::sliderChanged(int slider_id, float val)
{
    switch (slider_id)
    {
    case 0:
        if (!m_partb) {
            m_shrink_coeff = val;
            echo(m_shrink_coeff);
        }
        else {
            px = (val - 0.5) * 50;
            echo(px);
        }
        break;
    case 1:
        if (!m_partb) {
            m_inflate_coeff = -val;
            echo(m_inflate_coeff);
        }
        else {
            py = (val - 0.5) * 50;
            echo(py);
        }
        break;
    case 2:
        if (m_partb) {
            pz = (val - 0.5) * 50;
            echo(pz);
        }
        break;
    case 3:
        if (m_partb) {
            size = 5 * val;
            echo(size);
            // if model2 visible
            if (val == 0) val = 0.01;
            m_model2.setBigSize(val * getSceneWidth() / 2);
            m_model2.update();
        }
        break;
    case 4:
        if (m_partb) {
            if (val < 0.02) val = 0.02;
            m_resolution = 100 * val;
            echo(m_resolution);
        }
        break;
    }
}

void Mesh3DScene::draw()
{

	if (m_style_flag & FLAG_SHOW_SOLID) m_model.draw(m_obj_col, SOLID);
	if (m_style_flag & FLAG_SHOW_WIRE) m_model.draw(Colour::black, WIRE);
	if (m_style_flag & FLAG_SHOW_NORMALS) m_model.draw(Colour::black, NORMALS);
	if (m_style_flag & FLAG_SHOW_AXES) m_model.draw(Colour::black, AXES);

    if (m_partb && m_show_mesh2) {
        m_model2.centerAlign();
        m_model2.move(vec(px, py, pz));
        m_model2.update();

        if (m_style_flag & FLAG_SHOW_SOLID) m_model2.draw(m_obj_col, SOLID);
        if (m_style_flag & FLAG_SHOW_WIRE) m_model2.draw(Colour::black, WIRE);
        if (m_style_flag & FLAG_SHOW_NORMALS) m_model2.draw(Colour::black, NORMALS);
        if (m_style_flag & FLAG_SHOW_AXES) m_model2.draw(Colour::black, AXES);
    }

     if (m_style_flag & FLAG_SHOW_DIFF) {
         for (Point3D& p : m_points_diff) {
             p.draw();
         }
     }

     if (m_partb && m_show_lines) {
         for (LineSeg3D& l : m_lines) {
             l.draw();
         }
     }

     if (m_partb) {
         Point3D(px, py, pz).draw();
     }

     //for (vec& p : m_points_sample) {
     //    Point3D(p.x, p.y, p.z, Colour::white).draw();
     //}

     if (m_partb && m_show_sdf) {
         for (int i = 0; i < m_points_sample.size(); i++) {
             vec p = m_points_sample[i];
             vec f = 5 * force_field[i];
             vec v = p + f;
             if (!f.Equals(0, 0, 0)) {
                 LineSeg3D(p.x, p.y, p.z, v.x, v.y, v.z, Colour::orange).draw();
             }
             float d = distance_field[i];
             if (d > 0) {
                 Point3D(p.x, p.y, p.z, Colour::Colour(255 * (1 - d / distance_max_p), 255, 0)).draw();
             }
             else { // d <= 0
                 Point3D(p.x, p.y, p.z, Colour::Colour(255, 255 * (1 + d / distance_max_n), 0)).draw();
             }
         }
     }

     if (m_show_tri) {
         for (Triangle3D& tri : m_triangles3D) {
             tri.draw();
         }
     }

     if (m_partb) {
         LineSeg3D(m_force_point_of_application.x, m_force_point_of_application.y, m_force_point_of_application.z, m_force_point_of_application.x + m_force_vector.x, m_force_point_of_application.y + m_force_vector.y, m_force_point_of_application.z + m_force_vector.z, Colour::green).draw();
     }

     if (m_partb && m_show_sphere) {
         m_sphere = Sphere3D(px, py, pz, size, Colour::cyan);
         m_sphere.draw();
     }
     
     if (m_partb && m_show_box) {
         m_box = Box3D(px - size, py - size, pz - size, px + size, py + size, pz + size, Colour::cyan);
         m_box.setTransparency(0.5);
         m_box.draw();
     }
}

int main(int argc, char* argv[])
{
    try {
        return vvr::mainLoop(argc, argv, new Mesh3DScene);
    }
    catch (std::string exc) {
        cerr << exc << endl;
        return 1;
    }
    catch (...)
    {
        cerr << "Unknown exception" << endl;
        return 1;
    }
}

void SparseIdentity(SparseMatrix<float>& I, int n) {
    I.reserve(VectorXi::Constant(n, 1));
    for (int i = 0; i < n; i++) {
        I.insert(i, i) = 1;
    }
}

void SparseDiagonalInverse(SparseMatrix<float>& D, SparseMatrix<float>& D_inverse, int n) {
    for (int i = 0; i < n; i++) {
        D_inverse.coeffRef(i, i) = 1.0f / D.coeffRef(i, i);
    }
}

Plane FromTriangle(const vvr::Triangle& t)
{
    Plane result;
    result.normal = t.getNormal();
    result.d = Dot(result.normal, t.v1());
    return result;
}

float Raycast(const math::Plane& plane, const math::Ray& ray)
{
    float nd = Dot(ray.dir, plane.normal);
    float pn = Dot(ray.pos, plane.normal);

    // uncomment to check only normals facing the same way
    //if (nd >= 0.0f) {
    //    return -1;
    //}

    float t = (plane.d - pn) / nd;
    if (t > 0.0f) {
        return t;
    }

    return -1;
}

float Raycast(const vvr::Triangle& tri, const math::Ray& ray)
{
    Plane plane = FromTriangle(tri);
    float t = Raycast(plane, ray);
    if (t < 0.0f) {
        return t;
    }
    vec result = ray.pos + ray.dir * t;

    // convert result point to barycentric coordinates
    // https://math.stackexchange.com/a/28552
    double area = math::Cross(tri.v2() - tri.v1(), tri.v3() - tri.v1()).Length() / 2.0;
    double a = math::Cross(tri.v2() - result, tri.v3() - result).Length() / (2.0 * area);
    double b = math::Cross(tri.v3() - result, tri.v1() - result).Length() / (2.0 * area);
    double c = math::Cross(tri.v1() - result, tri.v2() - result).Length() / (2.0 * area);

    // test if result point is inside triangle using barycentric coordinates
    if (a > 0 && a < 1 && b > 0 && b < 1 && c > 0 && c < 1 && a + b + c > 0.99999 && a + b + c < 1.00001) {
        return t;
    }
    return -1;
}

vec ClosestPoint(const vvr::Triangle& t, const vec& p)
{
    // project point to triangle plane
    // https://stackoverflow.com/a/9605695
    vec n = t.getNormal();
    vec orig = t.v1();
    vec v = p - orig;
    double dist = Dot(v, n);
    vec pp = p - dist * n;

    // convert pp to barycentric coordinates
    // https://math.stackexchange.com/a/28552
    double area = math::Cross(t.v2() - t.v1(), t.v3() - t.v1()).Length() / 2.0;
    double a = math::Cross(t.v2() - pp, t.v3() - pp).Length() / (2.0 * area);
    double b = math::Cross(t.v3() - pp, t.v1() - pp).Length() / (2.0 * area);
    double c = math::Cross(t.v1() - pp, t.v2() - pp).Length() / (2.0 * area);

    // test if pp inside triangle using barycentric coordinates
    if (a > 0 && a < 1 && b > 0 && b < 1 && c > 0 && c < 1 && a + b + c > 0.99999 && a + b + c < 1.00001) {
        return pp;
    }
    else {
        double d_v1 = (pp - t.v1()).LengthSq();
        double d_v2 = (pp - t.v2()).LengthSq();
        double d_v3 = (pp - t.v3()).LengthSq();
        double d_v1v2 = (pp - t.v1()).LengthSq() - SQUARE(Dot((t.v2() - t.v1()).Normalized(), pp - t.v1()));
        double d_v2v3 = (pp - t.v2()).LengthSq() - SQUARE(Dot((t.v3() - t.v2()).Normalized(), pp - t.v2()));
        double d_v3v1 = (pp - t.v3()).LengthSq() - SQUARE(Dot((t.v1() - t.v3()).Normalized(), pp - t.v3()));

        if (Dot((t.v2() - t.v1()).Normalized(), pp - t.v1()) < 0 ||
            SQUARE(Dot((t.v2() - t.v1()).Normalized(), pp - t.v1())) > (t.v2() - t.v1()).LengthSq()) {
            d_v1v2 = 1e10;
        }
        if (Dot((t.v3() - t.v2()).Normalized(), pp - t.v2()) < 0 ||
            SQUARE(Dot((t.v3() - t.v2()).Normalized(), pp - t.v2())) > (t.v3() - t.v2()).LengthSq()) {
            d_v2v3 = 1e10;
        }
        if (Dot((t.v1() - t.v3()).Normalized(), pp - t.v3()) < 0 ||
            SQUARE(Dot((t.v1() - t.v3()).Normalized(), pp - t.v3())) > (t.v1() - t.v3()).LengthSq()) {
            d_v3v1 = 1e10;
        }

        if (d_v1 <= d_v2 && d_v1 <= d_v3 && d_v1 <= d_v1v2 && d_v1 <= d_v2v3 && d_v1 <= d_v3v1) {
            return t.v1();
        }
        else if (d_v2 <= d_v1 && d_v2 <= d_v3 && d_v2 <= d_v1v2 && d_v2 <= d_v2v3 && d_v2 <= d_v3v1) {
            return t.v2();
        }
        else if (d_v3 <= d_v1 && d_v3 <= d_v2 && d_v3 <= d_v1v2 && d_v3 <= d_v2v3 && d_v3 <= d_v3v1) {
            return t.v3();
        }
        else if (d_v1v2 <= d_v1 && d_v1v2 <= d_v2 && d_v1v2 <= d_v3 && d_v1v2 <= d_v2v3 && d_v1v2 <= d_v3v1) {
            vec v1v2 = t.v1() + Dot((t.v2() - t.v1()).Normalized(), pp - t.v1()) * (t.v2() - t.v1()).Normalized();
            return v1v2;
        }
        else if (d_v2v3 <= d_v1 && d_v2v3 <= d_v2 && d_v2v3 <= d_v3 && d_v2v3 <= d_v1v2 && d_v2v3 <= d_v3v1) {
            vec v2v3 = t.v2() + Dot((t.v3() - t.v2()).Normalized(), pp - t.v2()) * (t.v3() - t.v2()).Normalized();
            return v2v3;
        }
        else if (d_v3v1 <= d_v1 && d_v3v1 <= d_v2 && d_v3v1 <= d_v3 && d_v3v1 <= d_v1v2 && d_v3v1 <= d_v2v3) {
            vec v3v1 = t.v3() + Dot((t.v1() - t.v3()).Normalized(), pp - t.v3()) * (t.v1() - t.v3()).Normalized();
            return v3v1;
        }
    }

    return vec(0, 0, 0);
}

bool TriangleSphere(const vvr::Triangle& t, const vvr::Sphere3D& s)
{
    vec center = vec(s.x, s.y, s.z);
    vec closest = ClosestPoint(t, center);
    float magSq = closest.DistanceSq(center);
    return magSq <= SQUARE(s.rad);
}

Interval GetInterval(const vvr::Box3D& aabb, const vec& axis)
{
    vec vertex[8] = { // holds all the AABB vertices
        vec(aabb.x1, aabb.y1, aabb.z1),
        vec(aabb.x1, aabb.y1, aabb.z2),
        vec(aabb.x1, aabb.y2, aabb.z1),
        vec(aabb.x1, aabb.y2, aabb.z2),
        vec(aabb.x2, aabb.y1, aabb.z1),
        vec(aabb.x2, aabb.y1, aabb.z2),
        vec(aabb.x2, aabb.y2, aabb.z1),
        vec(aabb.x2, aabb.y2, aabb.z2)
    };

    Interval result;
    result.min = result.max = Dot(axis, vertex[0]);

    for (int i = 1; i < 8; i++) {
        float projection = Dot(axis, vertex[i]);
        result.min = (projection < result.min) ? projection : result.min;
        result.max = (projection > result.max) ? projection : result.max;
    }

    return result;
}

Interval GetInterval(const vvr::Triangle& t, const vec& axis)
{
    vec vertex[3] = { // holds all the triangle vertices
        t.v1(),
        t.v2(),
        t.v3()
    };

    Interval result;

    result.min = result.max = Dot(axis, vertex[0]);

    for (int i = 1; i < 3; i++) {
        float projection = Dot(axis, vertex[i]);
        result.min = (projection < result.min) ? projection : result.min;
        result.max = (projection > result.max) ? projection : result.max;
    }

    return result;
}

bool OverlapOnAxis(const vvr::Box3D& aabb, const vvr::Triangle& t, const vec& axis)
{
    Interval a = GetInterval(aabb, axis);
    Interval b = GetInterval(t, axis);
    return ((b.min <= a.max) && (a.min <= b.max));
}

bool TriangleAABB(const vvr::Triangle& t, const vvr::Box3D aabb)
{
    // find the edge vectors of the triangle
    vec f0 = t.v2() - t.v1();
    vec f1 = t.v3() - t.v2();
    vec f2 = t.v1() - t.v3();

    // find the face normals of the AABB
    vec u0(1, 0, 0);
    vec u1(0, 1, 0);
    vec u2(0, 0, 1);

    // potential separating axes
    vec test[13] = {
        u0, u1, u2, // AABB normals
        t.getNormal(), // triangle normal
        math::Cross(u0, f0), math::Cross(u0, f1), math::Cross(u0, f2), // cross product of every
        math::Cross(u1, f0), math::Cross(u1, f1), math::Cross(u1, f2), // normal of the AABB with
        math::Cross(u2, f0), math::Cross(u2, f1), math::Cross(u2, f2)  // every edge of the triangle
    };

    // test every axis to check for an overlap
    // if at least one is found, return false
    for (int i = 0; i < 13; i++) {
        if (!OverlapOnAxis(aabb, t, test[i])) {
            return false; // separating axis found
        }
    }

    return true; // separating axis not found
}

bool OverlapOnAxis(const vvr::Triangle& t1, const vvr::Triangle& t2, const vec& axis)
{
    Interval a = GetInterval(t1, axis);
    Interval b = GetInterval(t2, axis);
    return ((b.min <= a.max) && (a.min <= b.max));
}

bool TriangleTriangle(const vvr::Triangle& t1, const vvr::Triangle& t2)
{
    // edges of t1
    vec t1_f0 = t1.v2() - t1.v1();
    vec t1_f1 = t1.v3() - t1.v2();
    vec t1_f2 = t1.v1() - t1.v3();

    // edges of t2
    vec t2_f0 = t2.v2() - t2.v1();
    vec t2_f1 = t2.v3() - t2.v2();
    vec t2_f2 = t2.v1() - t2.v3();

    // potential separating axes
    vec test[11] = {
        t1.getNormal(), // t1 normal
        t2.getNormal(), // t2 normal
        math::Cross(t2_f0, t1_f0), math::Cross(t2_f0, t1_f1), math::Cross(t2_f0, t1_f2), // cross product of
        math::Cross(t2_f1, t1_f0), math::Cross(t2_f1, t1_f1), math::Cross(t2_f1, t1_f2), // every edge of t1 with
        math::Cross(t2_f2, t1_f0), math::Cross(t2_f2, t1_f1), math::Cross(t2_f2, t1_f2)  // every edge of t2
    };

    // test every axis to check for an overlap
    // if at least one is found, return false
    for (int i = 0; i < 11; i++) {
        if (!OverlapOnAxis(t1, t2, test[i])) {
            return false; // separating axis found
        }
    }

    return true; // separating axis not found
}

vec ClosestPoint(const vvr::Box3D& aabb, const vec& p)
{
    vec result = p;
    vec min = vec(aabb.x1, aabb.y1, aabb.z1);
    vec max = vec(aabb.x2, aabb.y2, aabb.z2);

    // closest point has to be at least min
    result.x = (result.x < min.x) ? min.x : result.x;
    result.y = (result.y < min.y) ? min.y : result.y;
    result.z = (result.z < min.z) ? min.z : result.z;

    // closest point has to be at most max
    result.x = (result.x > max.x) ? max.x : result.x;
    result.y = (result.y > max.y) ? max.y : result.y;
    result.z = (result.z > max.z) ? max.z : result.z;

    return result;
}

bool AABBSphere(const vvr::Box3D& aabb, const vvr::Sphere3D& s)
{
    vec center = vec(s.x, s.y, s.z);
    vec closest = ClosestPoint(aabb, center);
    float magSq = closest.DistanceSq(center);
    return magSq <= SQUARE(s.rad);
}

bool AABBAABB(const vvr::Box3D& aabb1, const vvr::Box3D& aabb2)
{
    // test for overlap on each world axis
    return (aabb1.x1 <= aabb2.x2 && aabb1.x2 >= aabb2.x1) &&
           (aabb1.y1 <= aabb2.y2 && aabb1.y2 >= aabb2.y1) &&
           (aabb1.z1 <= aabb2.z2 && aabb1.z2 >= aabb2.z1);
}

void ExportVector(const std::vector<float>& toExport, std::string path)
{
    ofstream fout;
    try {
        fout.open(path);
    }
    catch (int e) {
        cerr << "Error: Could not create file '" << path << "'!" << endl;
        return;
    }
    for (float value : toExport) {
        fout << value << endl;
    }
    fout.close();
    cout << "Exported to " << path << " successfully!" << endl;
}

void ExportVector(const std::vector<vec>& toExport, std::string path)
{
    ofstream fout;
    try {
        fout.open(path);
    }
    catch (int e) {
        cerr << "Error: Could not create file '" << path << "'!" << endl;
        return;
    }
    for (vec value : toExport) {
        fout << value.x << " " << value.y << " " << value.z << endl;
    }
    fout.close();
    cout << "Exported to " << path << " successfully!" << endl;
}

void ImportVector(std::vector<float>& toImport, std::string path) {
    ifstream fin;
    try {
        fin.open(path);
    }
    catch (int e) {
        cerr << "Error: Cound not access file '" << path << "'!" << endl;
        return;
    }

    toImport.clear();

    string line;
    while (getline(fin, line)) {
        stringstream lineStream(line);
        float value;
        lineStream >> value;
        toImport.push_back(value);
    }
    fin.close();

    cout << "Imported " << toImport.size() << " elements." << endl;
}

void ImportVector(std::vector<vec>& toImport, std::string path)
{
    ifstream fin;
    try {
        fin.open(path);
    }
    catch (int e) {
        cerr << "Error: Cound not access file '" << path << "'!" << endl;
        return;
    }

    toImport.clear();

    string line;
    while (getline(fin, line)) {
        stringstream lineStream(line);
        vec value;
        lineStream >> value.x >> value.y >> value.z;
        toImport.push_back(value);
    }
    fin.close();

    cout << "Imported " << toImport.size() << " elements." << endl;
}